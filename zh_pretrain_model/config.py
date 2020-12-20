#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/14 17:17
# @Author  : yangyang.clare
# @File    : config.py
# @contact: yang.a.yang@transwarp.io


class Config():
    def __init__(self):

        self.dataset = {
            'TNEWS':{
                'train':"data/raw_data/TNEWS_train1128.csv",
                'test':'data/raw_data/TNEWS_a.csv'
            },
            "OCNLI":{
                'train': "data/raw_data/OCNLI_train1128.csv",
                'test': 'data/raw_data/OCNLI_a.csv'
            },
            "OCEMOTION":{
                'train': "data/raw_data/OCEMOTION_train1128.csv",
                'test': 'data/raw_data/OCEMOTION_a.csv'
            }
        }
        # fine_tune
        self.pretrain_model = '/mnt/e/code/pretrain_models/bert_base_chinese'
        self.pretrain_model_path = 'state_dict/corpus_pretrain'
        self.fine_tnue_dropout = 0.1
        self.max_sequence_length = 128
        self.batch_size = 32
        self.learning_rate = 2e-5
        self.epochs = 10





