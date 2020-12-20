#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/21 17:36
# @Author  : yangyang.clare
# @File    : demo.py
# @contact: yang.a.yang@transwarp.io


from fine_tune.multi_task_features import *

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

def main():

    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.pretrain_model_path)

    processors = {
        'OCNLI':OCNLIProcess,
        'TNEWS':TNEWSProcess,
        'OCEMOTION':OCEMOTIONProcess
    }

    #load data
    oce_train_feas,oce_dev_feas,oce_test_feas = prepare_features('OCEMOTION',processors,config,tokenizer)
    ocn_train_feas,ocn_dev_feas,ocn_test_feas = prepare_features('OCNLI',processors,config,tokenizer)
    tnews_train_feas,tnews_dev_feas,tnews_test_feas = prepare_features('TNEWS',processors,config,tokenizer)



if __name__ == '__main__':
    main()
