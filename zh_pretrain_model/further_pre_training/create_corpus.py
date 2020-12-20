#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/18 15:42
# @Author  : yangyang.clare
# @File    : create_corpus.py
# @contact: yang.a.yang@transwarp.io

import os
import pandas as pd
import numpy as np
import copy,codecs

def get_filelist(dir):
    fullnames = []
    for home, dirs, files in os.walk(dir):
        for filename in files:
            fullname = os.path.join(home, filename)
            fullnames.append(fullname)
    return fullnames

fullnames = get_filelist('../data/raw_data')
texts = []
for name in fullnames:
    if name.startswith('OCNLI'):
        labels =['id','text','abs','label']
        if 'train' in name:
            data = pd.read_csv(name,sep='\t',names = labels)
        else:
            data = pd.read_csv(name, sep='\t', names=labels[:-1])
        data = data[['text','abs']].values.tolist()
        data = [line[0].replace('\t','') + '' + line[-1].replace('\t','') for line in data]

        texts += data

    else:
        labels = ['id','text','label']
        if 'train' in name:
            data = pd.read_csv(name, sep='\t', names=labels)
        else:
            data = pd.read_csv(name, sep='\t', names=labels[:-1])
        data = data['text'].values.tolist()

        texts += data
print(len(texts))
print(texts[:10])

with codecs.open('../data/corpus.txt','w') as f:
    for line in texts:
        f.write(line + '\n')
        f.write('\n')
# df = pd.DataFrame(texts,columns=['texts'])
# df.to_csv('../data/corpus.csv',index=False)