#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/16 16:12
# @Author  : yangyang.clare
# @File    : fine_tune_model.py
# @contact: yang.a.yang@transwarp.io

import pandas as pd
import numpy as np
import codecs,os
from zh_pretrain_game.utils.config import Config
from zh_pretrain_game.utils.process import *
from zh_pretrain_game.models.fine_tune import fine_tune

import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

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

if __name__ == '__main__':
    config = Config()
    texts = funetine_dataset(config)
    model, tokenizer = load_pretrain_model(config)

    if os.path.exists(config.persist['path']):
        # with open(config.persist['input_ids'],'rb') as f:
        #     input_ids = pickle.load(f)
        # with open(config.persist['masks'],'rb') as f:
        #     masks = pickle.load(f)
        with open(config.persist['input_ids'], 'rb') as out_data:
            # 按保存变量的顺序加载变量
            input_ids = pickle.load(out_data)
        with open(config.persist['masks'], 'rb') as out_data:
            # 按保存变量的顺序加载变量
            maks = pickle.load(out_data)
    else:
        os.makedirs(config.persist['path'])
        input_ids, masks = format_data(tokenizer, texts, config)
        with open(config.persist['input_ids'],'wb') as f:
            pickle.dump(input_ids,f,pickle.HIGHEST_PROTOCOL)
        with open(config.persist['masks'],'wb') as f:
            pickle.dump(masks,f,pickle.HIGHEST_PROTOCOL)

    print('>>>>>>>' * 5,'data process finished','>>>>' * 5)
    model = fine_tune(config.pretrain_model,config)
    fine_tune_model = fune_tune(input_ids, masks, model,config.batch_size)
    print('>>>>>>>' * 5, 'fine-tune finished', '>>>>' * 5)
    save_model(fine_tune_model,config.fine_tnue_model_path)
