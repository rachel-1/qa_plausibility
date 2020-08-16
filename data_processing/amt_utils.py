import boto3
import xmltodict
import requests
import pandas as pd
from custom_tokenizer import find_start_end
from tqdm import tqdm_notebook

class AMT:
    def __init__(self, production=False):
        environments = {
          "production": {
            "endpoint": "https://mturk-requester.us-east-1.amazonaws.com",
            "preview": "https://www.mturk.com/mturk/preview"
          },
          "sandbox": {
            "endpoint": 
              "https://mturk-requester-sandbox.us-east-1.amazonaws.com",
            "preview": "https://workersandbox.mturk.com/mturk/preview"
          }
        }
        self.mturk_environment = environments["production"] if production else environments["sandbox"]
        session = boto3.Session(profile_name='default')
        self.client = session.client(
            service_name='mturk',
            region_name='us-east-1',
            endpoint_url=self.mturk_environment['endpoint'],
        )
    
    def balance(self):
        return self.client.get_account_balance()['AvailableBalance']
    
    def create_hits(self, question_xml, task_attributes, df):
        count = 0
        total = len(df)
        print(df.index)
        # do one pair per HIT
        for pair in zip(df[::2].itertuples(), df[1::2].itertuples()):
            row_one, row_two = pair
            print(row_one.Index, row_two.Index)
            xml = question_xml
            xml = xml.replace('${question_1}',row_one.question)
            xml = xml.replace('${response_1}',row_one.response_filtered)
            xml = xml.replace('${question_2}',row_two.question)
            xml = xml.replace('${response_2}',row_two.response_filtered)
            
            # Add a URL to a base directory for images.
            img_url = "TODO"
            if requests.head(img_url+str(row_one.Index)+".jpg").status_code != requests.codes.ok:
                print("Image Not found:", row_one.Index)
                continue
            if requests.head(img_url+str(row_two.Index)+".jpg").status_code != requests.codes.ok:
                print("Image Not found:", row_two.Index)
                continue
                
            xml = xml.replace('${image_id_1}',str(row_one.Index))
            xml = xml.replace('${image_id_2}',str(row_two.Index))

            response = self.client.create_hit(
                **task_attributes,
                Question=xml
            )
            hit_type_id = response['HIT']['HITTypeId']
            df.loc[row_one.Index, 'hit_id'] = response['HIT']['HITId']
            df.loc[row_two.Index, 'hit_id'] = response['HIT']['HITId']
            df.loc[row_one.Index, 'hit_idx'] = '1'
            df.loc[row_two.Index, 'hit_idx'] = '2'
            
            count += 2
            print("Just created HIT {}, {}/{}".format(response['HIT']['HITId'], count, total))
        print("You can view the HITs here:")
        print(self.mturk_environment['preview']+"?groupId={}".format(hit_type_id))
        
    def generate_qualifying_task(self, df, example_indices=None):
        # https://docs.aws.amazon.com/AWSMechTurk/latest/AWSMturkAPI/ApiReference_QuestionFormDataStructureArticle.html
        def add_image(img_id):
            xml = "<EmbeddedBinary><EmbeddedMimeType><Type>image</Type><SubType>jpg</SubType></EmbeddedMimeType>"
            xml += "<DataURL>TODO"+str(img_id)+".jpg</DataURL>"
            xml += "<AltText>Image not found. Please contact the requester</AltText>"
            xml += "<Width>100</Width><Height>100</Height></EmbeddedBinary>"
            return xml
        
        def add_data(img_id, question, response):
            xml = "<Overview>"
            xml += add_image(img_id)
            xml += "<Text>Question: "+question+"</Text>"
            xml += "<Text>Response: "+response+"</Text></Overview>"
            return xml
            
        def add_q(identifier, display_name, question_text, is_example=False, true_answer=None, explanation=None):
            xml = "<Question><QuestionIdentifier>"+identifier+"</QuestionIdentifier>"
            xml += "<DisplayName>"+display_name+"</DisplayName>"
            xml += "<IsRequired>true</IsRequired>"
            xml += "<QuestionContent><Text>"+question_text+"</Text></QuestionContent>"
            # add possible answers
            xml += "<AnswerSpecification><SelectionAnswer><StyleSuggestion>radiobutton</StyleSuggestion>"
            xml += "<Selections>"
            for a in ['yes', 'no']:
                xml += "<Selection><SelectionIdentifier>"+a+"</SelectionIdentifier>"
                xml += "<Text>"+a
                xml += " [CORRECT ANSWER:{}]".format(explanation) if is_example and a == true_answer else ""
                xml += "</Text></Selection>"
            xml += "</Selections></SelectionAnswer></AnswerSpecification></Question>"
            return xml
        
        def add_answer(identifier, true_answer):
            xml = "<Question><QuestionIdentifier>"+identifier+"</QuestionIdentifier>"
            for a in ['yes', 'no']:
                xml += "<AnswerOption><SelectionIdentifier>"+a+"</SelectionIdentifier>"
                xml += "<AnswerScore>"+('1' if a == true_answer else '0')+"</AnswerScore></AnswerOption>"
            xml += "</Question>"
            return xml
        
        # link to magic AWS XML template things
        questions_xml = "<QuestionForm xmlns='http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionForm.xsd'>"
        answers_xml = "<AnswerKey xmlns='http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/AnswerKey.xsd'>"
        
        # add help text
        questions_xml += "<Overview><Text> The images were found on Instagram and the questions were asked of the posters. The questions were generated by a bot and the responses are free-text from social media users, so either one could be wrong. Answer the two questions for each image. The first few are simply examples. Make sure you understand each one, then click the option that says [CORRECT ANSWER].</Text></Overview>"

        # add questions and answers
        for idx, row in df.iterrows():
            is_example = example_indices is not None and idx in example_indices
            is_q_relevant = 'yes' if row.q_relevant else 'no'
            is_r_relevant = 'yes' if row.r_relevant else 'no'
            questions_xml += add_data(idx, row.question, row.response_filtered)
            questions_xml += add_q("q_relevant_{}".format(idx), "{}.{}".format(idx,1), "Is the question valid with respect to the image?", is_example=is_example, true_answer=is_q_relevant, explanation=row.get('q_relevant_explanation'))
            answers_xml += add_answer("q_relevant_{}".format(idx), is_q_relevant)
            questions_xml += add_q("r_relevant_{}".format(idx), "{}.{}".format(idx,2), "Is the response valid with respect to the image?", is_example=is_example, true_answer=is_r_relevant, explanation=row.get('r_relevant_explanation'))
            answers_xml += add_answer("r_relevant_{}".format(idx), is_r_relevant)
            
        # add method for calculating score    
        answers_xml += "<QualificationValueMapping><PercentageMapping>"
        answers_xml += "<MaximumSummedScore>"+str(len(df)*2)+"</MaximumSummedScore>"
        answers_xml += "</PercentageMapping></QualificationValueMapping>"

        # wrap up xml
        answers_xml += "</AnswerKey>"
        questions_xml += "</QuestionForm>"

        qualification = self.client.create_qualification_type(
                        Name='Question/Response Classification Task Understanding',
                        Keywords='test, qualification',
                        Description='This is a brief test to ensure workers understand the task set-up (half the "questions" are just examples)',
                        QualificationTypeStatus='Active',
                        RetryDelayInSeconds=60,
                        Test=questions_xml,
                        AnswerKey=answers_xml,
                        TestDurationInSeconds=300)

        return qualification['QualificationType']['QualificationTypeId']
    
    def get_reviewable_HITs(self):
        HITIds = []
        response = self.client.list_reviewable_hits()
        token = response.get('NextToken')
        HITIds.extend([HIT['HITId'] for HIT in response['HITs']])
        while(token is not None):
            response = self.client.list_reviewable_hits(NextToken=token)
            token = response.get('NextToken')
            HITIds.extend([HIT['HITId'] for HIT in response['HITs']])
        return HITIds
    
    def populate_results(self, df, ids=None):
        assert('hit_id' in df.columns)
        if ids is None:
            # skip rows that are already filled out
            if 'q_relevant' in df.columns and 'r_relevant' in df.columns:
                ids = list(df[pd.isnull(df.q_relevant) | pd.isnull(df.r_relevant)].hit_id.dropna().values)
                ids.extend(df[df.r_relevant & pd.isnull(df.turker_answer)].hit_id.dropna().values)
            else:
                ids = df.hit_id.dropna().values
        updated_HITs = []
        skipped_HITs = []
        for HITId in tqdm_notebook(ids):
            if HITId not in df.hit_id.values:
                print("HIT ID {} not found in df. Skipping...".format(HITId))
                continue
            
            # get a list of the Assignments that have been submitted
            assignmentsList = self.client.list_assignments_for_hit(
                HITId=HITId,
                MaxResults=10
            )
            if(len(assignmentsList['Assignments']) == 0): 
                skipped_HITs.append(HITId)
                continue
            assignment = assignmentsList['Assignments'][0]
            data_record_idx = df[df.hit_id == HITId].index.values
            data_record_idx = data_record_idx
            assert(df.at[data_record_idx[0],'hit_idx'] == 1)
            assert(df.at[data_record_idx[1],'hit_idx']== 2)
            df.at[data_record_idx, 'worker_id'] = assignment['WorkerId']
            df.at[data_record_idx, 'assignment_id'] = assignment['AssignmentId']
            answers = xmltodict.parse(assignment['Answer'])['QuestionFormAnswers']['Answer']
            answer_dict = {}
            for answer in answers:
                answer_dict[answer['QuestionIdentifier']] = answer['FreeText']
                
            df.at[data_record_idx[0], 'q_relevant'] = answer_dict['is-question-relevant-1'] == 'yes'
            df.at[data_record_idx[0], 'r_relevant'] = answer_dict['is-response-relevant-1'] == 'yes'
            if answer_dict['answer-1']: 
                df.at[data_record_idx[0], 'turker_answer'] = find_start_end(df.at[data_record_idx[0], 'r_tokenization'], answer_dict['answer-1'])
            df.at[data_record_idx[1], 'q_relevant'] = answer_dict['is-question-relevant-2'] == 'yes'
            df.at[data_record_idx[1], 'r_relevant'] = answer_dict['is-response-relevant-2'] == 'yes'
            if answer_dict['answer-2']: 
                df.at[data_record_idx[1], 'turker_answer'] = find_start_end(df.at[data_record_idx[1], 'r_tokenization'], answer_dict['answer-2'])
            
            updated_HITs.append(HITId)
        remaining_HITs = df[pd.notna(df.hit_id) & pd.isna(df.q_relevant)]
        print("{} HITs updated in df, {} skipped, {} remaining".format(len(updated_HITs), len(skipped_HITs), len(remaining_HITs)//2))
        return updated_HITs

    def approve_HITs(self, HITIds):
        for hit_id in HITIds:
            hit = self.client.get_hit(HITId=hit_id)
            assignmentsList = self.client.list_assignments_for_hit(
                              HITId=hit_id,
                              MaxResults=10
                              )
            for assignment in assignmentsList['Assignments']:
                if assignment['AssignmentStatus'] == 'Submitted':
                    self.client.approve_assignment(
                            AssignmentId=assignment['AssignmentId'],
                            OverrideRejection=False
                        )
                    
    def get_all_HITs(self, filter_fn=lambda x: True):
        HITIds = []
        response = self.client.list_hits()
        token = response.get('NextToken')
        HITIds.extend([HIT['HITId'] for HIT in response['HITs'] if filter_fn(HIT)])
        while(token is not None):
            response = self.client.list_hits(NextToken=token)
            token = response.get('NextToken')
            HITIds.extend([HIT['HITId'] for HIT in response['HITs'] if filter_fn(HIT)])
        return HITIds

    def expire_HITs(self, hitIDs):
        try:
            for hitId in hitIDs:
                self.client.update_expiration_for_hit(HITId=hitId, ExpireAt=0)
        except Exception as e:
                print(e)
                print("ID {} could not be expired.".format(hitId))

    def delete_HITs(self, hitIds):
        for hitId in hitIds:
            try:
                self.client.delete_hit(HITId=hitId)        
            except Exception as e:
                print(e)
                print("ID {} could not be deleted.".format(hitId))
                
    def give_qualification(self, qualificationID, workerIDs):
        try:
            for workerID in workerIDs:
                self.client.associate_qualification_with_worker(
                    WorkerId=workerID,
                    QualificationTypeId=qualificationID,
                    IntegerValue=100,
                    SendNotification=False)
        except Exception as e:
                print(e)
                print("ID {} could not be assigned.".format(workerID))
         
    def email_workers(self, workerIDs, email_params):
        self.client.notify_workers(**email_params, WorkerIds=workerIDs)