import pandas as pd
import os
import torch
import fire
from argparse import Namespace

from transformers import (BertConfig, BertForVQR, BertTokenizer)
from run_BERT import eval, get_eval_dataloader, VQRProcessor

def custom_eval(restore_file, dl_name='val', q_relevance=False, r_relevance=False, answer_extraction=False):
    config_file = os.path.join(os.path.dirname(restore_file), 'config.json')
    eval_path = restore_file[:-len(".bin")]+'_'+dl_name+'_eval.csv'
    print("Now generating new eval df...")

    args = Namespace(data_dir="data",
                     do_mini=('mini' in dl_name),
                     max_seq_length = 50,
                     eval_batch_size = 64,
                     q_relevance = q_relevance,
                     r_relevance = r_relevance,
                     answer_extraction = answer_extraction
    )

    config = BertConfig(config_file)
    processor = VQRProcessor(args.do_mini, args.q_relevance, args.r_relevance)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    model = BertForVQR(config, num_labels=num_labels, q_relevance=args.q_relevance, r_relevance=args.r_relevance, answer_extraction=args.answer_extraction)
    model.load_state_dict(torch.load(restore_file))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    if 'val' in dl_name:
        eval_examples = processor.get_dev_examples(args.data_dir)
        dl, token_mapping = get_eval_dataloader(args, eval_examples, label_list, tokenizer)
    elif 'test' in dl_name:
        print("Loading test set")
        test_examples = processor.get_test_examples(args.data_dir)
        dl, token_mapping = get_eval_dataloader(args, test_examples, label_list, tokenizer)

    df = eval(device, model, dl, args, save_results=True, original_examples=eval_examples, token_mappings=token_mapping)

    def convert_span(row):
        tok_to_orig = token_mapping[row.name]
        try:
            if row.raw_span_start <= row.raw_span_end:
                return (tok_to_orig[int(row.raw_span_start)], tok_to_orig[int(row.raw_span_end)])
            else:
                return None
        except Exception as e:
            print("tok_to_orig: ", tok_to_orig)
            print("row.raw_span_start: ", row.raw_span_start)
            print("row.raw_span_end: ", row.raw_span_end)
            print("Error converting span:", e)
            return None

    if args.q_relevance:
        df['q_prediction'] = df.q_prediction.apply(lambda x: True if x == 0 else False)
        
    if args.r_relevance:
        df['r_prediction'] = df.r_prediction.apply(lambda x: True if x == 0 else False)
        
    if args.answer_extraction:
        df['pred_span'] = df.apply(lambda x: convert_span(x), axis=1)

    # add in the rest of the info from the original data
    original_data = pd.read_csv(processor.full_path)
    df = df.join(original_data, rsuffix='r')
        
    df.to_csv(eval_path, index=False)
    print("Saving to", eval_path)

if __name__ == '__main__':
  fire.Fire(custom_eval)
