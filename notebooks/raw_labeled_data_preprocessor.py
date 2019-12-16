#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
# from flashtext import KeywordProcessor
# from bs4 import BeautifulSoup
# import requests
# from pypac import PACSession, get_pac
from tqdm import tqdm_notebook as tqdm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# import dask.dataframe as dd
# import fastText
# from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
import joblib
# import torch
# from torch import autograd
from os.path import isfile
import re
from time import time
from scipy import spatial
import random as rand
from random import random
import numpy as np
import math
import matplotlib.pyplot as plt
import json

tqdm().pandas()


# In[ ]:


import json
import codecs

new_labeled_data = json.load(codecs.open('../data/testData.json', 'r', 'utf-8-sig'))


# In[ ]:


new_labeled_data = new_labeled_data['rasa_nlu_data']['common_examples']


# In[ ]:


new_labeled_data[0]


# In[ ]:


intent_entity_dict = {}
for item in new_labeled_data:
    if item['intent'] not in intent_entity_dict:
        intent_entity_dict[item['intent']] = set()
    for entity in item['entities']:
        intent_entity_dict[item['intent']].add(entity['entity'])


# # 输出用于分类的数据

# In[ ]:


new_label_df = []
for t in new_labeled_data:
    new_label_df.append({
        'sentence': t['text'],
        'event': t['intent']
    })
    
new_label_df = pd.DataFrame(new_label_df)


# In[ ]:


new_label_df.head()


# In[ ]:


new_label_train, new_label_eval = train_test_split(new_label_df, test_size=0.3, stratify=new_label_df.event)


# In[ ]:


original_train_df = pd.read_csv("../data/event_type_entity_extract_train.csv", names=['id', 'sentence', 'event', 'answer'])


# In[ ]:


original_train_df['event'].unique()


# In[ ]:


updated_train_df = pd.concat([new_label_df, original_train_df], axis=0)


# In[ ]:


updated_train_df.to_csv("../temp/cat_classification_train.csv", index=False, header=False, columns=['id', 'sentence', 'event'])


# In[ ]:


new_label_eval.to_csv("../temp/cat_classification_eval.csv", index=False, header=False, columns=['id', 'sentence', 'event', 'answer'])


# # 输出用于抽取实体的数据

# In[ ]:


for t  in ['资金账户风险', '涉嫌欺诈', '业绩下滑', '信批违规', '涉嫌传销', '交易违规', '财务造假', '评级调整',
       '重组失败', '涉嫌违法', '实控人股东变更', '不能履职', '涉嫌非法集资', '资产负面', '歇业停业',
       '提现困难', '高管负面', '投诉维权', '失联跑路', '产品违规', '公司股市异常']:
    intent_entity_dict[t]={'主体'}


# In[ ]:


intent_entity_dict


# In[ ]:


joblib.dump(intent_entity_dict, "../temp/intent_entity_dict.pkl")


# In[ ]:


import random
new_label_df = []
for t in new_labeled_data:
    intent = t['intent']
    sentence = t['text']
    entity_map = {}
    contains_entity = set()
    for entity in t['entities']:
        new_label_df.append({
            'sentence': sentence,
            'query': intent+'__'+ entity['entity'],
            'answer': entity['value']
        })
        contains_entity.add(entity['entity'])
    contains_entity = [t for t in intent_entity_dict[intent] if t not in contains_entity]
    if contains_entity:
        for t in contains_entity:
            new_label_df.append({
                'sentence': sentence,
                'query': intent+'__'+ t,
                'answer': 'no_given'
            })
#     temp = list(intent_entity_dict.keys())
#     temp.remove(intent)
#     mock_intents = random.sample(temp, 6)
#     for mock_intent in mock_intents:
#         mock_entity = random.choice(list(intent_entity_dict[mock_intent]))
#         for mock_entity in intent_entity_dict[mock_intent]:
#             new_label_df.append({
#                 'sentence': sentence,
#                 'query': mock_intent+'__'+ mock_entity,
#                 'answer': 'no_given'
#             })


# In[ ]:


new_label_df = pd.DataFrame(new_label_df)


# In[ ]:


new_label_df.shape


# In[ ]:


new_label_df['id'] = new_label_df.index


# In[ ]:


new_label_df.head()


# In[ ]:


new_label_df.shape


# ## read original data

# In[ ]:


new_label_train, new_label_eval = train_test_split(new_label_df, test_size=0.3, stratify= new_label_df['query'])


# In[ ]:


original_train_df = pd.read_csv("../data/event_type_entity_extract_train.csv", names=['id', 'sentence', 'query', 'answer'])


# In[ ]:


original_train_df.shape


# In[ ]:


original_train_df['query'] = original_train_df['query'].apply(lambda x: x+'__主体' if x !='其他' else x)


# In[ ]:


original_train_df['query'].unique()


# In[ ]:


updated_train_df = pd.concat([new_label_df, original_train_df], axis=0)


# In[ ]:


updated_train_df.to_csv("../temp/cat_event_extraction_train.csv", index=False, header=False, columns=['id', 'sentence', 'query', 'answer'])


# In[ ]:


new_label_eval.to_csv("../temp/cat_event_extraction_eval.csv", index=False, header=False, columns=['id', 'sentence', 'query', 'answer'])


# In[ ]:


len(updated_train_df)


# In[ ]:


new_label_eval.shape


# In[ ]:




