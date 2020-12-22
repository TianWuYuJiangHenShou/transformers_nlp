#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/21 17:36
# @Author  : yangyang.clare
# @File    : demo.py
# @contact: yang.a.yang@transwarp.io


from fine_tune.multi_task_features import *
from config import Config
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def prepare_features(task_name,processors,config,tokenizer):
    if task_name == 'OCEMOTION':
        examples,labels2id = processors[task_name]().get_train_examples(config.dataset[task_name]['train'])
        train,dev = processors[task_name]().get_dev_examples(examples)
        test = processors[task_name]().get_test_examples(config.dataset[task_name]['test'])
    else:
        examples = processors[task_name]().get_train_examples(config.dataset[task_name]['train'])
        train, dev = processors[task_name]().get_dev_examples(examples)
        test = processors[task_name]().get_test_examples(config.dataset[task_name]['test'])

    train_features = convert_examples_to_features(train,config.max_sequence_length,tokenizer)
    dev_features = convert_examples_to_features(dev, config.max_sequence_length, tokenizer)
    test_features = convert_examples_to_features(test, config.max_sequence_length, tokenizer)
    return train_features,dev_features,test_features

def convert_feature_to_tensor(features):
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    return input_ids,input_mask,segment_ids,label_ids

def build_train_dataLoader(input_ids,input_mask,segment_ids,label_ids,bs):
    data = TensorDataset(input_ids, input_mask, segment_ids,label_ids)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=bs)
    return dataloader

def build_test_dataLoader(input_ids,input_mask,segment_ids,bs):
    data = TensorDataset(input_ids, input_mask, segment_ids)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=bs)
    return dataloader

def _train(config,tokenizer,processors,task_name):
    train_feas,dev_feas,test_feas = prepare_features(task_name,processors,config,tokenizer)
    train_ids,train_masks,train_seg_ids,train_labels= convert_feature_to_tensor(train_feas)
    dev_ids,dev_masks,dev_seg_ids,dev_labels= convert_feature_to_tensor(dev_feas)
    test_ids,test_masks,test_seg_ids,_= convert_feature_to_tensor(test_feas)

    train_dataloader = build_train_dataLoader(train_ids,train_masks,train_seg_ids,train_labels,config.batch_size)
    dev_dataloader = build_train_dataLoader(dev_ids,dev_masks,dev_seg_ids,dev_labels,config.batch_size)
    test_dataloader = build_test_dataLoader(test_ids,test_masks,test_seg_ids,config.batch_size)

def _evaluate():
    pass


def _inference():
    pass


def main():

    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.pretrain_model_path)

    processors = {
        'OCNLI':OCNLIProcess,
        'TNEWS':TNEWSProcess,
        'OCEMOTION':OCEMOTIONProcess
    }



if __name__ == '__main__':
    main()
