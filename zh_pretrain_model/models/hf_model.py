#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/17 15:51
# @Author  : yangyang.clare
# @File    : base_model.py
# @contact: yang.a.yang@transwarp.io

from transformers import BertModel,BertConfig
from torch import nn
import torch
from models.squeeze_embedding import SqueezeEmbedding

class SequenceClassfication(nn.Module):

    def __init__(self,config,data_type):
        super(SequenceClassfication, self).__init__()
        self.num_labels = config.label_nums[data_type]
        bert_config = BertConfig.from_pretrained(config.pretrain_model_path)
        bert_config.num_labels = self.num_labels
        self.bert_config = bert_config

        self.bert = BertModel.from_pretrained(config.pretrain_model_path)
        self.linear = nn.Linear(self.bert_config.hidden_size,self.bert_config.num_labels)
        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.squeezeembedding = SqueezeEmbedding()

    def forward(self,input_ids,attention_masks,segment_ids):

        lengths = torch.sum(input_ids != 0, dim=-1)
        #加快训练速度，但可能丢失精度
        input_ids = self.squeezeembedding(input_ids,lengths)
        sequence_output = self.bert(input_ids,attention_masks,segment_ids)

        output = sequence_output[1]
        output = self.dropout(output)
        logits = self.linear(output)

        return logits






