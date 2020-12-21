#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/21 17:36
# @Author  : yangyang.clare
# @File    : demo.py
# @contact: yang.a.yang@transwarp.io


from fine_tune.multi_task_features import *
from config import Config
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup,BertTokenizer
from models.label_smooth import CrossEntropyLoss_LSR
from models.old_model import SequenceClassfication
from tqdm import trange
import tqdm,os
import numpy as np
from torch.nn import CrossEntropyLoss

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

def convert_feature_to_tensor(features,set_type = 'train'):
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if set_type == 'train':
        label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        return input_ids,input_mask,segment_ids,label_ids
    else:
        return input_ids,input_mask,segment_ids

def build_train_dataLoader(input_ids,input_mask,segment_ids,label_ids,bs):
    data = TensorDataset(input_ids, input_mask, segment_ids,label_ids)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=bs)
    return dataloader

def build_test_dataLoader(input_ids,input_mask,segment_ids,bs):
    data = TensorDataset(input_ids, input_mask, segment_ids)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=bs)
    return dataloader

def _train(config,tokenizer,processors,task_name):

    train_feas,dev_feas,test_feas = prepare_features(task_name,processors,config,tokenizer)
    train_ids,train_masks,train_seg_ids,train_labels= convert_feature_to_tensor(train_feas)
    dev_ids,dev_masks,dev_seg_ids,dev_labels= convert_feature_to_tensor(dev_feas)
    test_ids,test_masks,test_seg_ids= convert_feature_to_tensor(test_feas,set_type='test')

    train_dataloader = build_train_dataLoader(train_ids,train_masks,train_seg_ids,train_labels,config.batch_size)
    dev_dataloader = build_train_dataLoader(dev_ids,dev_masks,dev_seg_ids,dev_labels,config.batch_size)
    test_dataloader = build_test_dataLoader(test_ids,test_masks,test_seg_ids,config.batch_size)

    logger.info('load data success')

    model = SequenceClassfication(config,task_name)
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_parameters)

    epochs = config.epochs
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    loss_fct = CrossEntropyLoss_LSR(device,para_LSR=0.1)
    # loss_fct = CrossEntropyLoss()
    for epoch in trange(epochs,desc = 'train epochs of {}'.format(task_name)):
        model.train()
        train_loss,train_steps = 0.0,0
        for step,batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids,mask,segment_id,labels = batch
            logits = model(input_ids,mask,segment_id)
            loss = loss_fct(logits,labels)

            loss.backward()
            train_loss += loss.item()
            train_steps += 1

            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)

            scheduler.step()
            optimizer.step()

            if step % 50 == 0:
                print('epoch :{},step:{},train loss:{}'.format(epoch,step,train_loss / train_steps))

        model.eval()
        dev_loss,dev_steps = 0.0,0
        dev_acc,dev_examples = 0.0,0
        for batch in dev_dataloader:
            batch = tuple(t.to(device) for t in batch)
            eval_id,eval_mask,eval_seg,eval_label = batch

            with torch.no_grad():
                eval_logits = model(eval_id,eval_mask,eval_seg)

            eval_loss = loss_fct(eval_logits,eval_label)

            eval_out = eval_logits.detach().cpu().numpy()
            eval_label = eval_label.detach().cpu().numpy()
            outputs = np.argmax(eval_out,axis= -1)
            batch_acc = np.sum(outputs == eval_label)

            dev_loss += eval_loss.item()
            dev_steps += 1

            dev_acc += batch_acc
            dev_examples += eval_id.size(0)

        dev_acc = dev_acc / dev_examples
        dev_loss = dev_loss / dev_steps

        print('epoch:{},dev_acc:{},dev_loss:{}')
    torch.save(model.state_dict(), './state_dict/fine_tune_models/{}.pt'.format(task_name), _use_new_zipfile_serialization=False)


def main():

    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.pretrain_model_path)

    processors = {
        'OCNLI':OCNLIProcess,
        'TNEWS':TNEWSProcess,
        'OCEMOTION':OCEMOTIONProcess
    }

    for task_name in processors.keys():
        _train(config,tokenizer,processors,task_name)


if __name__ == '__main__':
    main()
