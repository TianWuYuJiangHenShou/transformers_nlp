#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/14 16:48
# @Author  : yangyang.clare
# @File    : process.py
# @contact: yang.a.yang@transwarp.io

import pandas as pd
import numpy as np
import codecs,os
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
    input_ids,masks = [] ,[]
    # CLS,SEP = tokenizer.convert_tokens_to_ids(['[CLS]']),tokenizer.convert_tokens_to_ids(['[SEP]'])
    for line in texts:
        tokens = tokenizer.encode_plus(line.strip().replace('\n',''),max_length=config.max_sequence_length,pad_to_max_length = True)
        input_id, token_ids, mask = tokens['input_ids'],tokens['token_type_ids'],tokens['attention_mask']
        input_ids.append(input_id)
        masks.append(mask)
    return input_ids,masks

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


if __name__ == '__main__':
    config = Config()
    texts = funetine_dataset(config)
    model, tokenizer = load_pretrain_model(config)

    if os.path.exists(config.persist['path']):
        with open(config.persist['input_ids'],'wb') as f:
            input_ids = pickle.load(f)
        with open(config.persist['masks'],'wb') as f:
            masks = pickle.load(f)
    else:
        os.makedirs(config.persist['path'])
        input_ids, masks = format_data(tokenizer, texts, config)
        with open(config.persist['input_ids'],'wb') as f:
            pickle.dump(input_ids,f)
        with open(config.persist['masks'],'wb') as f:
            pickle.dump(masks,f)

    print('>>>>>>>' * 5,'data process finished','>>>>' * 5)

    fine_tune_model = fune_tune(input_ids, masks, model,config.batch_size)
    print('>>>>>>>' * 5, 'fine-tune finished', '>>>>' * 5)
    save_model(fine_tune_model,config.fine_tnue_model_path)
