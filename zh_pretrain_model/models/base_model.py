#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/17 15:51
# @Author  : yangyang.clare
# @File    : base_model.py
# @contact: yang.a.yang@transwarp.io

# from transformers import BertModel,BertConfig
from torch import nn
import torch,os
from models.squeeze_embedding import SqueezeEmbedding
from fine_tune.modeling import BertConfig,BertModel
class SequenceClassfication(nn.Module):

    def __init__(self,config,data_type,lstm_hidden_size):
        super(SequenceClassfication, self).__init__()
        self.num_labels = config.label_nums[data_type]
        # bert_config = BertConfig.from_pretrained(config.pretrain_model_path)
        # bert_config.num_labels = self.num_labels
        # self.bert_config = bert_config
        bert_config = BertConfig.from_json_file(os.path.join(config.pretrain_model_path, 'bert_config.json'))

        self.lstm_hidden_size = lstm_hidden_size
        self.bert = BertModel(bert_config)
        self.bert_config = bert_config
        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.squeezeembedding = SqueezeEmbedding()

        self.lstm = nn.LSTM(self.bert_config.hidden_size, lstm_hidden_size,
                            num_layers=1, bidirectional=True, dropout=0.3, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(self.lstm_hidden_size,100),
            nn.LayerNorm(100),
            nn.Linear(100,self.num_labels)
        )
        self.linear = nn.Linear(lstm_hidden_size, self.num_labels)

    def forward(self,input_ids,attention_masks,segment_ids):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        lengths = torch.sum(input_ids != 0, dim=-1)
        #加快训练速度，但可能丢失精度
        # input_ids = self.squeezeembedding(input_ids,lengths)
        sequence_output = self.bert(input_ids,attention_masks,segment_ids)

        embeds = sequence_output[0]
        #取倒数第二层
        lstm_out,hidden = self.lstm(embeds[-2])
        lstm_out = lstm_out.contiguous().view(-1, self.lstm_hidden_size * 2)
        output = self.dropout(lstm_out)

        output = output.view(batch_size,seq_len,-1)
        values,indexs = output.topk(1)

        values = values.view(batch_size,-1)
        logits = self.fc(values)

        return logits






