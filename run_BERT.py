# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import json
from datetime import datetime
import pandas as pd
import scipy
import sklearn.metrics

from tensorboardX import SummaryWriter

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, BertForVQR, BertTokenizer)
from transformers import AdamW, WarmupLinearSchedule

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None,
                 q_relevant=None, r_relevant=None,
                 answer=None, span=None, id=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = id
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.q_relevant = q_relevant
        self.r_relevant = r_relevant
        self.answer = answer
        self.span = span

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()
    
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class VQRProcessor(DataProcessor):
    """Processor for custom VQR data set."""

    def __init__(self, mini=False, q_relevance=False, r_relevance=False, train_frac=1):
        self.mini = mini
        self.q_relevance = q_relevance
        self.r_relevance = r_relevance
        self.train_frac = train_frac

    def get_examples(self, dataset_name, data_dir):
        if self.q_relevance and self.r_relevance:
            prefix = 'q+r_relevance_'
        elif self.q_relevance:
            prefix = 'q_relevance_'
        else:
            prefix = 'r_relevance_'

        self.full_path = os.path.join(data_dir, prefix+dataset_name+'.csv')
        logger.info("LOOKING AT {}".format(self.full_path))
        return self._create_examples(self.full_path, dataset_name)
        
    def get_train_examples(self, data_dir):
        """See base class."""
        dataset_name = 'train' if not self.mini else 'minitrain'
        if self.train_frac != 1:
            dataset_name += '_'+str(self.train_frac)
        return self.get_examples(dataset_name, data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        dataset_name = 'val' if not self.mini else 'minival'
        return self.get_examples(dataset_name, data_dir)

    def get_test_examples(self, data_dir):
        """See base class."""
        dataset_name = 'test'
        return self.get_examples(dataset_name, data_dir)
    
    def get_labels(self):
        """See base class."""
        return [True, False]

    def _create_examples(self, full_path, set_type):
        """Creates examples for the training and dev sets."""
        with open(full_path, 'r') as f:
            dataset = csv.DictReader(f)
            examples = []
            for (i, vals) in enumerate(dataset):
                example = {}
                example['guid'] = "%s-%s" % (set_type, i)
                example['id'] = i
                example['text_a'] = vals['question']
                example['text_b'] = vals['response_filtered'].split(' ')
                from ast import literal_eval
                if 'q_relevant' in vals:
                    example['q_relevant'] = literal_eval(vals['q_relevant'])
                if 'r_relevant' in vals:
                    example['r_relevant'] = literal_eval(vals['r_relevant'])
                if 'answer_intersection_span' in vals:
                    example['span'] = literal_eval(vals['answer_intersection_span']) if vals['answer_intersection_span'] != '' else None
                examples.append(InputExample(**example))
        return examples
        
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    token_mapping = {}
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            orig_to_tok_start_index = []
            orig_to_tok_end_index = []
            tok_to_orig_index = []#*list(range(len(tokens_a)+2))]
            tokens_b = []
            for (i, token) in enumerate(example.text_b):
                sub_tokens = tokenizer.tokenize(token)
                if len(sub_tokens) == 0: sub_tokens = [tokenizer.wordpiece_tokenizer.unk_token]
                orig_to_tok_start_index.append(len(tokens_b))
                orig_to_tok_end_index.append(len(tokens_b)+len(sub_tokens)-1)
                for sub_token in sub_tokens:
                    tokens_b.append(sub_token)
                    tok_to_orig_index.append(i)

            if example.span is not None and example.span != '':
                ans_start_idx = orig_to_tok_start_index[example.span[0]]
                ans_end_idx = orig_to_tok_end_index[example.span[1]]
            else:
                # these are just placeholders; their value will be ignored
                ans_start_idx, ans_end_idx = -1, -1

            # Trim tokens_b if necessary
            if len(tokens_a) + len(tokens_b) + len(['[CLS]', '[SEP]', '[SEP]']) > max_seq_length:
                orig_gt = tokens_b[ans_start_idx: ans_end_idx+1] if (example.span is not None) else None
                window_size = max_seq_length - 3 - len(tokens_a)

                # Take a window from the response such that the answer is contained
                if ans_start_idx == -1:
                    # since the -1 was just a placeholder, leaving these values is fine
                    ans_start_idx = np.random.randint(0, len(tokens_b))
                    ans_end_idx = ans_start_idx

                # truncate answer if it is too long
                if ans_end_idx - ans_start_idx + 1 >= window_size:
                    ans_end_idx = ans_start_idx + window_size - 1
                    gt_end = tok_to_orig_index[ans_end_idx]
                else:
                    gt_end = example.span[1] if example.span is not None else None
                    
                # calculate index of start of window
                lower_bound = max(ans_end_idx-window_size+1, 0)
                upper_bound = min(len(tokens_b)-window_size,ans_start_idx)
                window_start = np.random.randint(lower_bound, upper_bound+1)
                tokens_b = tokens_b[window_start:window_start+window_size]
                ans_start_idx -= window_start
                ans_end_idx -= window_start

                tok_to_orig_index = tok_to_orig_index[window_start:window_start+window_size]
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            if example.span is not None:
                ans_start_idx += len(tokens)
                ans_end_idx += len(tokens)
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        q_relevant_label_id = label_map.get(example.q_relevant)
        r_relevant_label_id = label_map.get(example.r_relevant)
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if example.q_relevant is not None:
                logger.info("q_relevant: %s (id = %d)" % (example.q_relevant, q_relevant_label_id))
            if example.r_relevant is not None:
                logger.info("r_relevant: %s (id = %d)" % (example.r_relevant, r_relevant_label_id))

        if not example.r_relevant:
            ans_start_idx = ans_end_idx = -1

        features.append(
                InputFeatures(example_id=example.example_id,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              q_relevance_label_id=q_relevant_label_id,
                              r_relevance_label_id=r_relevant_label_id,
                              ans_start_idx=ans_start_idx,
                              ans_end_idx=ans_end_idx))
        token_mapping[ex_index] = tok_to_orig_index
    return features, token_mapping

# Convert model output from a list to a dict
def list_to_dict(args, output):
    retval = {}
    keys = ['loss']
    if args.q_relevance: keys += ['q_logits']
    if args.r_relevance: keys += ['r_logits']
    if args.answer_extraction: keys += ['span_logits']
    count = 0
    for key in sorted(keys):
        retval[key] = output[count]
        count += 1
    return retval

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    num_removed_from_b = 0
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            num_removed_from_b += 1
    return num_removed_from_b

def accuracy(out, labels):
    predictions = np.argmax(out, axis=1)
    return np.sum(predictions==labels)/labels.size

def auroc(outputs, labels):
    scores = outputs[:, 1]
    try: roc = sklearn.metrics.roc_auc_score(labels, scores)
    except Exception as e:
        print(e)
        roc = 0 # will fail if all labels in a batch are the same
    return roc

def compute_exact(a_gold, a_pred):
    return int(a_gold == a_pred)

def f1(true_spans, pred_spans):
    f1 = 0
    count = 0
    for (true_span, pred_span) in zip(true_spans, pred_spans):
        if true_span[0] == -1: continue # placeholder
        count += 1
        pred_tokens = range(pred_span[0], pred_span[1]+1)
        if len(pred_tokens) == 0: continue
        true_tokens = range(true_span[0], true_span[1]+1)
        true_tokens = set(true_tokens)
        num_same = len(true_tokens.intersection(pred_tokens))
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(true_tokens)
        if precision + recall != 0:
            f1 += (2 * precision * recall) / (precision + recall)

    return f1, count
        
def train(device, model, optimizer, train_dataloader, args, eval_dataloader=None, tb_writer=None, original_train_examples=None, train_token_mappings=None, original_eval_examples=None, eval_token_mappings=None):
    global_step = 0
    best_eval_metric_val = 0
    model.train()
    for epoch_idx in trange(int(args.num_train_epochs), desc="Epoch"):
        metrics = {metric:0 for metric in ['loss', 'q_relevance_auroc', 'q_relevance_accuracy', 'r_relevance_auroc', 'r_relevance_accuracy']}
        metrics['raw_span_f1'] = [0, 0] # special case for keeping track of the count as well
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            
            # extract relevant info from the batch
            example_ids, input_ids, input_mask, segment_ids = batch[:4]
            q_relevance_label_ids, r_relevance_label_ids = None, None
            start_pos_ids, end_pos_ids = None, None
            count = 4
            if args.q_relevance:
                q_relevance_label_ids = batch[count]
                count += 1
            if args.r_relevance:
                r_relevance_label_ids = batch[count]
                count += 1
            if args.answer_extraction:
                start_pos_ids, end_pos_ids = batch[count:]

            original_ex_batch = [(original_train_examples[id], train_token_mappings[id]) for id in example_ids.cpu().numpy().tolist()]
            
            # evaluate the model (which will ignore inputs based on its configuration)
            results = model(input_ids, segment_ids, input_mask,
                            q_relevance_label_ids, r_relevance_label_ids,
                            start_pos_ids, end_pos_ids, original_examples=original_ex_batch)
            
            # extract info from the results
            results = list_to_dict(args, results)
            loss = results['loss']
            if args.q_relevance:
                q_logits = results['q_logits'].detach().cpu()
                q_relevance_label_ids = q_relevance_label_ids.cpu().numpy()
                metrics['q_relevance_auroc'] += auroc(q_logits, q_relevance_label_ids)
                metrics['q_relevance_accuracy'] += accuracy(q_logits.numpy(), q_relevance_label_ids)
            if args.r_relevance:
                r_logits = results['r_logits'].detach().cpu()
                r_relevance_label_ids = r_relevance_label_ids.cpu().numpy()
                metrics['r_relevance_auroc'] += auroc(r_logits, r_relevance_label_ids)
                metrics['r_relevance_accuracy'] += accuracy(r_logits.numpy(), r_relevance_label_ids)
            if args.answer_extraction:
                start_logits, end_logits = results['span_logits']
                start_pos_ids = start_pos_ids.cpu().numpy()
                end_pos_ids = end_pos_ids.cpu().numpy()
                start_logits = start_logits.detach().cpu()
                end_logits = end_logits.detach().cpu()
                start_idx = np.argmax(start_logits, axis=1)
                end_idx = np.argmax(end_logits, axis=1)
                raw_f1_score, count = f1(zip(start_pos_ids, end_pos_ids), zip(start_idx, end_idx))
                metrics['raw_span_f1'] = [metrics['raw_span_f1'][0] + raw_f1_score, metrics['raw_span_f1'][1] + count]
                
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            metrics['loss'] += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, AdamW handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        if eval_dataloader is not None:
            eval_metric_val = eval(device, model, eval_dataloader, args, epoch_idx, tb_writer, original_examples=original_eval_examples, token_mappings=eval_token_mappings)

        for metric_name, metric_value in metrics.items():
            if metric_value == 0: continue
            if metric_name != 'raw_span_f1':
                tb_writer.add_scalar(metric_name+'/train', metric_value / nb_tr_steps, epoch_idx)
            else:
                if metric_value[1] != 0:
                    tb_writer.add_scalar('span_f1/train', metric_value[0] / metric_value[1], epoch_idx)

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
        output_model_file = os.path.join(args.output_dir, "chkpt_{}.bin".format(epoch_idx))
        torch.save(model_to_save.state_dict(), output_model_file)

        # save new best model
        if eval_dataloader is not None:
            if eval_metric_val > best_eval_metric_val:
                best_eval_metric_val = eval_metric_val
                output_model_file = os.path.join(args.output_dir, "best.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
        
        # only save config once
        if epoch_idx == 0:
            output_config_file = os.path.join(args.output_dir, "config.json")
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())


def eval(device, model, eval_dataloader, args, iter_num=None, tb_writer=None, save_results=False, original_examples=None, token_mappings=None):
    model.eval()
    metrics = {metric:0 for metric in ['loss', 'q_relevance_auroc', 'q_relevance_accuracy', 'r_relevance_auroc', 'r_relevance_accuracy']}
    metrics['raw_span_f1'] = [0, 0] # special case for keeping track of the count as well
    nb_eval_steps, nb_eval_examples = 0, 0
    if save_results: df = pd.DataFrame()
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)

        # extract relevant info from the batch
        example_ids, input_ids, input_mask, segment_ids = batch[:4]
        q_relevance_label_ids, r_relevance_label_ids = None, None
        start_pos_ids, end_pos_ids = None, None
        count = 4
        if args.q_relevance:
            q_relevance_label_ids = batch[count]
            count += 1
        if args.r_relevance:
            r_relevance_label_ids = batch[count]
            count += 1
        if args.answer_extraction:
            start_pos_ids, end_pos_ids = batch[count:]

        original_ex_batch = [(original_examples[id], token_mappings[id]) for id in example_ids.cpu().numpy().tolist()]
            
        with torch.no_grad():
            # evaluate the model (which will ignore inputs based on its configuration)
            results = model(input_ids, segment_ids, input_mask,
                            q_relevance_label_ids, r_relevance_label_ids,
                            start_pos_ids, end_pos_ids, original_examples=original_ex_batch)
            
        # extract info from the results
        results = list_to_dict(args, results)
        loss = results['loss']
        if args.q_relevance:
            q_logits = results['q_logits'].detach().cpu()
            q_relevance_label_ids = q_relevance_label_ids.cpu().numpy()
            metrics['q_relevance_auroc'] += auroc(q_logits, q_relevance_label_ids)
            metrics['q_relevance_accuracy'] += accuracy(q_logits.numpy(), q_relevance_label_ids)
        if args.r_relevance:
            r_logits = results['r_logits'].detach().cpu()
            r_relevance_label_ids = r_relevance_label_ids.cpu().numpy()
            metrics['r_relevance_auroc'] += auroc(r_logits, r_relevance_label_ids)
            metrics['r_relevance_accuracy'] += accuracy(r_logits.numpy(), r_relevance_label_ids)
        if args.answer_extraction:
            start_logits, end_logits = results['span_logits']
            start_pos_ids = start_pos_ids.cpu().numpy()
            end_pos_ids = end_pos_ids.cpu().numpy()
            start_logits = start_logits.detach().cpu()
            end_logits = end_logits.detach().cpu()
            start_idx = np.argmax(start_logits, axis=1)
            end_idx = np.argmax(end_logits, axis=1)
            raw_f1_score, count = f1(zip(start_pos_ids, end_pos_ids), zip(start_idx, end_idx))
            metrics['raw_span_f1'] = [metrics['raw_span_f1'][0] + raw_f1_score, metrics['raw_span_f1'][1] + count]

        if save_results:
            response_starts = []
            for idx in range(segment_ids.shape[0]):
                response_start = (segment_ids[idx] == 1).nonzero()[0].item()
                response_starts.append(response_start)
            response_starts = torch.tensor(response_starts)
            idx_batch = list(range(nb_eval_examples, nb_eval_examples+input_ids.size(0)))
            cols = []
            data = np.array([]).reshape(input_ids.size(0), 0)
            if args.q_relevance:
                q_prediction = np.expand_dims(np.argmax(q_logits, axis=1).numpy(), 1)
                data = np.hstack([data, q_logits.numpy(), q_prediction])
                cols.extend(['q_score_0', 'q_score_1', 'q_prediction'])
            if args.r_relevance:
                r_prediction = np.expand_dims(np.argmax(r_logits, axis=1).numpy(), 1)
                data = np.hstack([data, r_logits.numpy(), r_prediction])
                cols.extend(['r_score_0', 'r_score_1', 'r_prediction'])
            if args.answer_extraction:
                start_idx -= response_starts
                end_idx -= response_starts
                data = np.hstack([data, np.expand_dims(start_idx,1), np.expand_dims(end_idx,1)])
                cols.extend(['raw_span_start', 'raw_span_end'])
            batch_df = pd.DataFrame(data, index=idx_batch, columns=cols)
            df = df.append(batch_df)
            
        metrics['loss'] += loss.mean().item()
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    for metric_name, metric_value in metrics.items():
        if tb_writer is not None:
            if metric_value == 0: continue
            if metric_name != 'raw_span_f1':
                tb_writer.add_scalar(metric_name+'/val', metric_value / nb_eval_steps, iter_num)
            else:
                tb_writer.add_scalar('span_f1/val', metric_value[0] / metric_value[1], iter_num)
            
    if save_results: return df
    return metrics['loss']

def get_eval_dataloader(args, eval_examples, label_list, tokenizer):
    eval_features, token_mapping = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)
    all_example_ids = torch.tensor([f.example_id for f in eval_features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    tensors = [all_example_ids, all_input_ids, all_input_mask, all_segment_ids]
    if args.q_relevance:
        # add label ids (question is confused or not)
        tensors.append(torch.tensor([f.q_relevance_label_id for f in eval_features], dtype=torch.long))
    if args.r_relevance:
        # add label ids (response is confused or not)
        tensors.append(torch.tensor([f.r_relevance_label_id for f in eval_features], dtype=torch.long))
    if args.answer_extraction:
        # add start/stop indices of answer
        tensors.append(torch.tensor([f.ans_start_idx for f in eval_features], dtype=torch.long))
        tensors.append(torch.tensor([f.ans_end_idx for f in eval_features], dtype=torch.long))
    eval_data = TensorDataset(*tensors)
    eval_sampler = SequentialSampler(eval_data)
    return DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size), token_mapping

def get_train_dataloader(train_features, args):
    tensors = []
    all_example_ids = torch.tensor([f.example_id for f in train_features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    tensors = [all_example_ids, all_input_ids, all_input_mask, all_segment_ids]
    if args.q_relevance:
        # add label ids (question is confused or not)
        tensors.append(torch.tensor([f.q_relevance_label_id for f in train_features], dtype=torch.long))
    if args.r_relevance:
        # add label ids (response is confused or not)
        tensors.append(torch.tensor([f.r_relevance_label_id for f in train_features], dtype=torch.long))
    if args.answer_extraction:
        # add start/stop indices of answer
        tensors.append(torch.tensor([f.ans_start_idx for f in train_features], dtype=torch.long))
        tensors.append(torch.tensor([f.ans_end_idx for f in train_features], dtype=torch.long))
    train_data = TensorDataset(*tensors)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)

    return DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="data",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model",
                        default="bert-base-uncased",
                        type=str, 
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--saved_model",
                        default=None,
                        type=str,
                        help="Fine-tuned model to load weights from")

    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        help="File to load config from")
    
    ## Other parameters
    parser.add_argument("--tensorboard_logdir",
                        default="runs",
                        type=str,
                        required=False,
                        help="The output directory where the tensorboard event files are saved.")
    parser.add_argument("--cache_dir",
                        default="~/local_model_cache",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=50,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_frac",
                        default=1.0,
                        type=float,
                        help="What percentage of the training data to use")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.",
                        default=True)
    parser.add_argument("--q_relevance",
                        action='store_true',
                        help="Whether to classify questions as confused or not.")
    parser.add_argument("--r_relevance",
                        action='store_true',
                        help="Whether to classify responses as confused or not.")
    parser.add_argument("--answer_extraction",
                        action='store_true',
                        help="Whether to extract answers")
    parser.add_argument("--answer_verification",
                        action='store_true',
                        help="Whether to verify answers",
                        default=False)
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.",
                        default=True)
    parser.add_argument("--do_mini",
                        action='store_true',
                        help="Whether not to mini version of the data")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--attention_dropout",
                        default=0.1,
                        type=float,
                        help="Percent dropout at attention layers")
    parser.add_argument("--hidden_dropout",
                        default=0.1,
                        type=float,
                        help="Percent dropout at hidden layers")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    log_dir = os.path.join(args.tensorboard_logdir, datetime.now().strftime('%Y-%m-%d--%H-%M-%S') + '_' + os.path.basename(args.output_dir[:-1] if args.output_dir[-1] == '/' else args.output_dir))
    os.mkdir(log_dir)
    fh = logging.FileHandler(log_dir+'/run.log')
    fh.setLevel(logging.DEBUG)

    tb_writer = SummaryWriter(logdir=log_dir)
    
    def get_free_gpu():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        os.remove('tmp')
        return np.argmax(memory_available)
        
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(f"cuda:{get_free_gpu()}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.output_dir = args.output_dir
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = VQRProcessor(args.do_mini, args.q_relevance, args.r_relevance, args.train_frac)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(np.ceil(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps)) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        train_features, train_token_mappings = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        train_dataloader = get_train_dataloader(train_features, args)
            
    if args.do_eval:
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_dataloader, eval_token_mappings = get_eval_dataloader(args, eval_examples, label_list, tokenizer)
        
    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))

    if args.saved_model is not None:
        print("Now loading from", args.saved_model)

        # Load a trained model and config that you have fine-tuned
        config = BertConfig(args.config_file)
        model = BertForVQR(config, num_labels=2, binary_only=args.binary_only, answer_extraction_only=args.answer_extraction_only, answer_verification=args.answer_verification)
        model.load_state_dict(torch.load(args.saved_model))
    else:
        config = BertConfig.from_pretrained(args.config_file)
        model = BertForVQR.from_pretrained(args.bert_model, cache_dir=cache_dir, config=config, num_labels=num_labels, q_relevance=args.q_relevance, r_relevance=args.r_relevance, answer_extraction=args.answer_extraction, answer_verification=args.answer_verification)
        
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        train(device, model, optimizer, train_dataloader, args, eval_dataloader, tb_writer, train_examples, train_token_mappings, eval_examples, eval_token_mappings)

    if not args.do_train and args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval(device, model, eval_dataloader, args)
    tb_writer.close()

if __name__ == "__main__":
    main()
