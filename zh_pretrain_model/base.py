#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/17 15:30
# @Author  : yangyang.clare
# @File    : base.py
# @contact: yang.a.yang@transwarp.io

# from transformers import BertTokenizer, BertForSequenceClassification,BertModel
import torch
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from fine_tune.modeling import BertModel,BertConfig

# config = BertConfig.from_json_file('state_dict/corpus_pretrain_gpu/bert_config.json')
# BertModel(config)

# tokenizer = BertTokenizer.from_pretrained('/mnt/e/code/pretrain_models/bert_base_chinese')
# model = BertForSequenceClassification.from_pretrained('/mnt/e/code/pretrain_models/bert_base_chinese')
# inputs = tokenizer("的就是打开", return_tensors="pt")
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
# outputs = model(**inputs, labels=labels)
# print(outputs)
# model =  BertModel.from_pretrained('/mnt/e/code/pretrain_models/bert_base_chinese')
# inputs = tokenizer("砂石款", return_tensors="pt")
# outputs = model(**inputs)
# print(type(outputs))
# model.state_dict()

data = pd.read_csv('data/raw_data/TNEWS_train1128.csv',sep='\t',names=['id','text','label'])
labels = data['label'].values.tolist()
print(set(labels))
print(len(set(labels)))
print([i for i in range(100,105)])





