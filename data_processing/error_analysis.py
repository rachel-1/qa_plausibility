from IPython.display import Image, display
import pandas as pd
import os
from custom_tokenizer import find_start_end
from tqdm import tqdm_notebook
import numpy as np

# Add the directory for images (listed as {id}.jpg)
IMAGE_BASE_DIR = 'TODO'

def validate_turker_responses(df, csv_file, target_num=None, worker_ids=None):
    if 'gold_answer_span' not in df.columns:
        df['gold_answer_span'] = None
    else:
        df['gold_answer_span'] = df.gold_answer.astype(object)
    
    if worker_ids is None: worker_ids = df.worker_id.dropna().unique()
    
    for worker_id in worker_ids:
        worker_specific_df = df[df.worker_id == worker_id]
    
        print("Worker ID: {} ({} total)".format(worker_id, len(worker_specific_df)))

        if target_num is None:
            target_num = min(3, len(worker_specific_df)//10+1)
   
        # skip ones that already have some manual eval
        if 'gold_q_relevant' in df.columns and len(worker_specific_df[pd.notna(worker_specific_df.gold_q_relevant)]) > target_num:
            continue

        if len(worker_specific_df) > target_num:
            worker_specific_df = worker_specific_df.sample(target_num)

        def display_func(row):
            print(f"{row.question} [{row.q_relevant}] {row.response_filtered} [{row.r_relevant}, {row.turker_answer}]")
        view_examples(worker_specific_df, df, csv_file, display_func=display_func)

def view_examples(subset_df, full_df, df_path, num_rows=50, unlabeled_only=True, skip_prompt=False,
                 display_func=None):
    """
    View individual rows from |subset_df|, allowing the user to update records in |full_df| by writing out to |df_path|.
    """  
    # check if gold columns in full_df
    if 'gold_answer_span' not in full_df.columns:
        full_df['gold_answer_span'] = None
    
    # skip labeled examples
    if unlabeled_only and 'gold_q_relevant' in full_df.columns:
        subset_df = subset_df.drop(full_df[pd.notnull(full_df.gold_q_relevant)].index, errors='ignore')
        
    for index, row in subset_df.head(num_rows).iterrows():
        image_name = str(index)+'.jpg'
        image_path = os.path.join(IMAGE_BASE_DIR, image_name)
        if os.path.isfile(image_path):
            display(Image(filename=image_path, width=200, height=200))
        else:
            raise Exception("Image not found at '{}'".format(image_path))
        
        if display_func: display_func(row)
        
        if skip_prompt: continue
        if (input("Modify record? (Y/N):").lower() == 'y'):
            q_label = input("Is q relevant? (Y/N):").lower()
            r_label = input("Is r relevant? (Y/N):").lower()
            answer = input("Extract answer:") if r_label != 'n' else None
        else:
            q_label = 'y' if row.q_relevant else 'n'
            r_label = 'y' if row.r_relevant else 'n'
            answer = row.answer_intersection
        if index in full_df.index:
            full_df.loc[index, 'gold_q_relevant'] = True if q_label == 'y' else False
            full_df.loc[index, 'gold_r_relevant'] = True if r_label == 'y' else False
            full_df.at[index, 'gold_answer_span'] = find_start_end(row.response_filtered.split(), answer) if answer is not None else None
        else:
            row['gold_q_relevant'] = (q_label == 'y')
            row['gold_r_relevant'] = (r_label == 'y')
            row['gold_answer_span'] = find_start_end(row.response_filtered.split(), answer) if answer is not None else None
            full_df = full_df.append(row)
        if df_path is not None: 
            cols = set(full_df.columns.values) - set(['score', 'prediction'])
            full_df.to_csv(df_path, index_label='index', columns=cols)
            print("Just wrote to file")
            
def calc_f1(true_span, pred_span):
    if true_span is None or pred_span is None: return 0
    pred_tokens = range(pred_span[0], pred_span[1]+1)
    if len(pred_tokens) == 0: return 0
    true_tokens = range(true_span[0], true_span[1]+1)
    if len(true_tokens) == 0: return 0
    true_tokens = set(true_tokens)
    num_same = len(true_tokens.intersection(pred_tokens))
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(true_tokens)
    if precision + recall == 0: return 0
    return (2 * precision * recall) / (precision + recall)

def exact_match(true_span, pred_span):
    if pred_span is None or true_span is None: return None
    return int(list(true_span) == list(pred_span))

def analyze_df(df, df_path, full_df=None, threshold=0.5, display_func=None):
    calc_metrics(df)
    if full_df is None: full_df = df
    view_examples(df, full_df, df_path, skip_prompt=False, unlabeled_only=True, display_func=display_func)

def calc_metrics(df):
    import sklearn.metrics
    total_num = len(df)
    
    # compute binary metrics
    def calc_binary_metrics(prefix):
        if not prefix in ['q', 'r']: 
            raise Exception("Prefix invalid")
        predicted = prefix+'_prediction'
        ground_truth = prefix+'_relevant'
        tp_df = df[(df[predicted] == True) & (df[ground_truth] == True)]
        fp_df = df[(df[predicted] == True) & (df[ground_truth] == False)]
        fn_df = df[(df[predicted] == False) & (df[ground_truth] == True)]
        tn_df = df[(df[predicted] == False) & (df[ground_truth] == False)]
        
        precision = len(tp_df)/(len(tp_df)+len(fp_df)) if len(tp_df) > 0 else float('nan')
        recall = len(tp_df)/(len(tp_df)+len(fn_df)) if len(tp_df) > 0 else float('nan')
        accuracy = (len(tp_df) + len(tn_df))/total_num
        f1 = (2*precision*recall)/(precision+recall)
        df_with_pred = df[pd.notnull(df[predicted])]
        auroc = sklearn.metrics.roc_auc_score(df_with_pred[ground_truth], df_with_pred[prefix+'_score_0'])
        print(predicted+" Precision: {:0.2f}, Recall: {:0.2f}, Accuracy: {:0.2f}, F1: {:0.2f}, AUROC: {:0.2f}".format(precision, recall, accuracy, f1, auroc))
            
    if 'q_prediction' in df.columns: calc_binary_metrics('q')
    if 'r_prediction' in df.columns: calc_binary_metrics('r')

    # compute span extraction metrics
    if 'predicted_answer' in df.columns:
        df['exact_match'] = df.apply(lambda row: exact_match(row['answer_intersection_span'], row['pred_span']), axis=1)
        em_score = np.mean(df[df.r_relevant].exact_match)
        df['f1'] = df.apply(lambda row: calc_f1(row.answer_intersection_span, row.pred_span) if row.r_relevant else False, axis=1)
        f1 = np.mean(df[df.r_relevant].f1)
        print("EM: {:0.2f}, F1: {:0.2f}".format(em_score, f1))
        