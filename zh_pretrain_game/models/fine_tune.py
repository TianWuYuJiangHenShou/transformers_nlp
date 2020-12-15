#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/16 15:58
# @Author  : yangyang.clare
# @File    : fine_tune.py
# @contact: yang.a.yang@transwarp.io

import torch
from torch import nn
from transformers import BertModel

class fine_tune(nn.Module):

    def __init__(self,path,config):
        super(fine_tune, self).__init__()
        self.bert = BertModel.from_pretrained(path)
        self.dropout = nn.Dropout(config.fine_tnue_dropout)


    def forward(self, input_ids,masks):
        output = self.bert(input_ids,masks)
        pool_output = output[1]
        pool_output = self.dropout(pool_output)
        return pool_output





