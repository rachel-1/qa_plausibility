# Determining Question-Answer Plausibility in Crowdsourced Datasets Using Multi-Task Learning
This repo provides the source code for the paper "Determining Question-Answer Plausibility in Crowdsourced Datasets Using Multi-Task Learning": 

Datasets extracted from social networks and online forums are often prone to the pitfalls of natural language, namely the presence of unstructured and noisy data. In this work, we seek to enable the collection of high-quality question-answer datasets from social media by proposing a novel task for automated quality analysis and data cleaning: question-answer (QA) plausibility. Given a machine or user-generated question and a crowd-sourced response from a social media user, we determine if the question and response are valid; if so, we identify the answer within the free-form response. We design BERT-based models to perform the QA plausibility task, and we evaluate the ability of our models to generate a clean, usable question-answer dataset. Our highest-performing model consists of a single-task model which determines the plausibility of the question, followed by a multi-task model which evaluates the plausibility of the response as well as extracts answers (Question Plausibility AUROC=0.75, Response Plausibility AUROC=0.78, Answer Extraction F1=0.665).

![Image of Pipeline](diagram.png)

## Setup

### 0. Python Environment
To load our dependencies in conda, you can run
```
conda env create -f environment.yml
conda activate qa_plausibility
```
You can also view the file directly and install dependencies on your own.

### 1. BERT Code
First clone the BERT code from the HuggingFace Transformers repo.
```
git clone https://github.com/huggingface/transformers
```
The safest option is to use the exact commit we started from (around October 2019), as shown below. Don't worry if it gives you warnings about whitespace.
```
cd transformers
git checkout -b qa_plausibility https://github.com/huggingface/transformers 5b6cafb11b39e78724dc13b57b81bd73c9a66b49
git apply ../qa_plausibility/modify_bert.diff
pip install -e .
```
If you would like a later version, but you can leave off the commit SHA, but you may need to resolve merge conflicts yourself.

### 2. Data Acquisition
While we are not releasing our data at this time due to privacy concerns, we do provide extensive documentation on our process for labeling, filtering and validating our data. The [README in our data_processing folder](data_processing/README.md) has details on how to use the provided Jupyter notebooks in that folder.

## Training and Evaluating
To train and evaluate the BERT model, you would run a command such as
```
python run_BERT.py --data_dir YOUR_DATA --do_train --do_eval --q_relevance --r_relevance --answer_extraction --output_dir OUTPUT_DIR --config_file config.json
```
You can provide any subset of `--q_relevance`, `--r_relevance` and `--answer_extraction` to perform any subset of tasks. Note that the code currently expects data files to be named according to `{task}_{dataset}.csv`. For example, the command above would expect to find `q+r_relevance_train.csv` and `q+r_relevance_val.csv` as output by the dataset splitting code given in `data_processing`. For more configuration, you can view all options using `python run_BERT.py -h`.

To do a more rigorous analysis of the outputs of a model, you can use
```
python evaluate_BERT.py MODEL_PATH --q_relevance --r_relevance --answer_extraction
```
to output an `{model_name}_eval.csv` file which shows the model predictions alongside the original inputs. To analyze metrics, view results and manually update ground truth labels as necessary, you can use the [Model Error Analysis](data_processing/7%20-%20Model%20Error%20Analysis.ipynb) notebook.
