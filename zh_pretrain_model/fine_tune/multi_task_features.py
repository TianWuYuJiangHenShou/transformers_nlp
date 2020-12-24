#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/21 14:36
# @Author  : yangyang.clare
# @File    : run_multi_tasks_classifier.py
# @contact: yang.a.yang@transwarp.io

from transformers import BertTokenizer,BertModel
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from config import Config

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DataProcessor(object):

    def get_train_examples(self,data_dir):
        raise NotImplementedError()

    def get_dev_examples(self,data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls,input_file,quotechar = None):

        with open(input_file,'r') as f:
            reader = csv.reader(f,delimiter = '\t',quotechar = quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class OCEMOTIONProcess(DataProcessor):

    def get_train_examples(self,data_dir):
        train_data = pd.read_csv(data_dir,sep='\t',header=None).values
        return self._create_examples(train_data,'train')

    def get_dev_examples(self,examples):
        train_examples, dev_examples = train_test_split(examples ,test_size=0.1)
        return train_examples,dev_examples

    def get_test_examples(self,data_dir):
        test_data = pd.read_csv(data_dir,sep='\t',header=None).values
        return self._create_examples(test_data,'test')

    def _create_examples(self,lines,set_type):
        examples = []
        labels2ids = {}
        for i,line in enumerate(lines):
            guid = '%s-%s' % (set_type,i)
            text_a = line[1]
            if set_type == 'train':
                label = line[-1]
                if label not in labels2ids:
                    num = len(labels2ids)
                    labels2ids[label] = num+1
                    label = labels2ids[label]
            else:
                label = None
            examples.append(
                InputExample(guid = guid,text_a = text_a,text_b = None,label = label)
            )
        if set_type == 'train':
            return examples,labels2ids
        else:
            return examples


class TNEWSProcess(DataProcessor):

    def get_train_examples(self,data_dir):
        train_data = pd.read_csv(data_dir,sep='\t',header=None).values
        return self._create_examples(train_data,'train')

    def get_dev_examples(self,examples):
        train_examples, dev_examples = train_test_split(examples ,test_size=0.1)
        return train_examples,dev_examples

    def get_test_examples(self,data_dir):
        test_data = pd.read_csv(data_dir,sep='\t',header=None).values
        return self._create_examples(test_data,'test')

    def _create_examples(self,lines,set_type):
        examples = []
        for i,line in enumerate(lines):

            guid = '%s-%s' % (set_type,i)
            text_a = line[1]
            if set_type == 'train':
                try:
                    label = int(line[-1])
                    if label in range(100,105):
                        label = label - 100
                    elif label in range(106,117):
                        label = label - 102
                except:
                    pass
            else:
                label = None
            examples.append(
                InputExample(guid = guid,text_a = text_a,text_b = None,label = label)
            )
        return examples


class OCNLIProcess(DataProcessor):

    def get_train_examples(self, data_dir):
        train_data = pd.read_csv(data_dir, sep='\t', header=None).values
        return self._create_examples(train_data, 'train')

    def get_dev_examples(self, examples):
        train_examples, dev_examples = train_test_split(examples, test_size=0.1)
        return train_examples, dev_examples

    def get_test_examples(self, data_dir):
        test_data = pd.read_csv(data_dir, sep='\t', header=None).values
        return self._create_examples(test_data, 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = '%s-%s' % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            if set_type == 'train':
                label = int(line[-1])
            else:
                label = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

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

def convert_examples_to_features(examples,max_seq_len,tokenizer):

    features = []

    for idx,example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)


        if tokens_b:
            _truncate_seq_pair(tokens_a,tokens_b,max_seq_len - 3)
        else:
            if len(tokens_a) > max_seq_len - 2:
                tokens_a = tokens_a[0:max_seq_len - 2]

        #build features
        tokens,segment_ids = [],[]
        tokens.append('[CLS]')
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append('[SEP]')
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append('[SEP]')
            segment_ids.append(1)

        assert len(tokens) == len(segment_ids)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        assert len(input_ids) == len(segment_ids)

        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # print(idx,len(input_ids),len(input_mask),len(segment_ids),tokens[:10])
        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        label_id = example.label

        if idx < 5:
            logger.info('***show examples***')
            logger.info('guid:%s' % (example.guid))
            logger.info('tokens: %s' % ' '.join([x for x in tokens]))
            logger.info('input_ids: %s' % ' '.join(str(x) for x in input_ids))
            logger.info('input_mask: %s' % ' '.join(str(x) for x in input_mask))
            logger.info('segments_ids: %s' % ' '.join(str(x) for x in segment_ids))
            logger.info('labels: %s ' % (example.label))

        features.append(
            InputFeatures(
                input_ids = input_ids,
                input_mask= input_mask,
                segment_ids=segment_ids,
                label_id=label_id
            )
        )
    return features













