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
                'train':"../data/dataset/TNEWS_train1128.csv",
                'test':'../data/dataset/TNEWS_a.csv'
            },
            "OCNLI":{
                'train': "../data/dataset/OCNLI_train1128.csv",
                'test': '../data/dataset/OCNLI_a.csv'
            },
            "OCEMOTION":{
                'train': "../data/dataset/OCEMOTION_train1128.csv",
                'test': '../data/dataset/OCEMOTION_a.csv'
            }
        }
        self.pretrain_model = '/mnt/e/code/pretrain_models/bert_base_chinese'
        self.fine_tnue_model_path = '../state_dict/fine_tune_model.pt'
        self.max_sequence_length = 64
        self.batch_size = 32

        #data Persistence
        self.persist = {
            'path':'../data/persistence',
            'input_ids':'../data/persistence/input_ids.txt',
            'masks':'../data/persistence/masks.txt'
        }


