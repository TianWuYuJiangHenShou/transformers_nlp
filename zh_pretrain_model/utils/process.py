#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/14 16:48
# @Author  : yangyang.clare
# @File    : process.py
# @contact: yang.a.yang@transwarp.io

import pandas as pd
import numpy as np
import codecs,os
from .config import Config
from transformers import BertTokenizer,BertModel,BertConfig
import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def load_data(path,columes):
    data = pd.read_csv(path, sep='\t',names = columes)
    return data

def funetine_dataset(config):
    texts = []
    for k,v in config.dataset.items():
        if k == 'OCNLI':
            train = load_data(config.dataset[k]['train'],columes=['id','sen1','sen2','label'])
            train['text'] = train['sen1'] + train['sen2']
            texts += train['text'].values.tolist()

            test = load_data(config.dataset[k]['test'],columes=['id','sen1','sen2'])
            test['text'] = test['sen1'] + test['sen2']
            texts += test['text'].values.tolist()

        else:
            train = load_data(config.dataset[k]['train'], columes=['id', 'sen', 'label'])
            train['text'] = train['sen']
            texts += train['text'].values.tolist()

            test = load_data(config.dataset[k]['test'], columes=['id', 'sen'])
            test['text'] = test['sen']
            texts += test['text'].values.tolist()

    return texts

def load_pretrain_model(config):
    pretrain_model = config.pretrain_model
    config = BertConfig.from_pretrained(os.path.join(pretrain_model,'config.json'))
    tokenizer = BertTokenizer.from_pretrained(pretrain_model)
    model = BertModel.from_pretrained(pretrain_model,config=config)

    return model,tokenizer


def format_data(tokenizer,texts,config):
    input_ids,masks,token_ids = [] ,[],[]
    # CLS,SEP = tokenizer.convert_tokens_to_ids(['[CLS]']),tokenizer.convert_tokens_to_ids(['[SEP]'])
    for line in texts:
        tokens = tokenizer.encode_plus(line.strip().replace('\n',''),max_length=config.max_sequence_length,pad_to_max_length = True)
        input_id, token_id, mask = tokens['input_ids'],tokens['token_type_ids'],tokens['attention_mask']
        input_ids.append(input_id)
        token_ids.append(token_id)
        masks.append(mask)
    return input_ids,masks,token_ids

def fune_tune(input_ids,masks,model,batch_size):

    input_ids = torch.Tensor(input_ids).long()
    masks = torch.Tensor(masks).long()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    input_ids, masks, model = input_ids.to(device), masks.to(device), model.to(device)

    train_data = TensorDataset(input_ids, masks)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    for step, batch in enumerate(train_dataloader):
        input_ids, masks = (i for i in batch)
        fine_tune_model = model(input_ids ,masks)
    return fine_tune_model

def save_model(model,save_path):
    torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)

def multi_task_process(config,tokenizer):
    # dataset = {
    #     'OCNLI':{
    #         'input_ids':[],
    #         'token_ids':[],
    #         'masks':[]
    #     },
    # ...
    # }
    dataset = {}
    for type,path in config.dataset.items():
        if type == 'OCNLI':
            input_ids,masks,token_ids = [] ,[],[]

            train = load_data(config.dataset[type]['train'], columes=['id', 'sen1','sen2', 'label'])
            sen_a = train['sen1'].values.tolist()
            sen_b = train['sen2'].values.tolist()

            for i,_ in enumerate(sen_a):
                tokens_a = tokenizer.encode_plus(sen_a.strip().replace('\n',''))
                tokens_b = tokenizer.encode_plus(sen_b.strip().replace('\n',''))
                # input_id, token_id, mask = tokens_a['input_ids'],tokens_a['token_type_ids'],tokens_a['attention_mask']
                text = tokens_a['input_ids'] + tokens_b['input_ids'][1:]
                token_type_id = tokens_a['token_type_ids'] + tokens_b['token_type_ids'][1:]
                mask = tokens_a['attention_mask'] + tokens_b['attention_mask'][1:]

                assert len(text) == len(token_type_id)
                assert len(text) == len(mask)

                if len(text) > config.max_sequence_length:
                    text,token_type_id,mask = text[:config.max_sequence_length],token_type_id[:config.max_sequence_length],mask[:config.max_sequence_length]
                else:
                    text = text + [tokenizer.pad_token_id] *(config.max_sequence_length - len(text))
                    token_type_id = token_type_id + [0] *(config.max_sequence_length - len(token_type_id))
                    mask = mask + [0] *(config.max_sequence_length - len(mask))

                assert len(text) == config.max_sequence_length
                assert len(token_type_id) == config.max_sequence_length
                assert len(mask) == config.max_sequence_length

                input_ids.append(text)
                token_ids.append(token_type_id)
                masks.append(mask)

        else:
            train = load_data(config.dataset[type]['train'], columes=['id', 'sen', 'label'])
            lines = train['sen'].values.tolist()
            input_ids,masks,token_ids = format_data(tokenizer,lines,config)
        dics = {}
        dics['input_ids'] = input_ids
        dics['token_ids'] = token_ids
        dics['masks'] = masks
        dataset[type] = dics
    return dataset

