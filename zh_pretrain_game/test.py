#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/16 17:02
# @Author  : yangyang.clare
# @File    : test.py
# @contact: yang.a.yang@transwarp.io

import pickle
# dataList = [[1, 1, 'yes'],
#             [1, 1, 'yes'],
#             [1, 0, 'no'],
#             [0, 1, 'no'],
#             [0, 1, 'no']]
# dataDic = { 0: [1, 2, 3, 4],
#             1: ('a', 'b'),
#             2: {'c':'yes','d':'no'}}
# with open('demo.txt','wb') as in_data:
#     pickle.dump(dataList,in_data,pickle.HIGHEST_PROTOCOL)
#     pickle.dump(dataDic,in_data,pickle.HIGHEST_PROTOCOL)
with open('data/persistence/input_ids.txt','rb') as out_data:
    # 按保存变量的顺序加载变量
    data = pickle.load(out_data)
    print(data) # dataList
    # data=pickle.load(out_data)
    # print(data) # dataDic
