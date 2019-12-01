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


new_label_df = []
for t in new_labeled_data:
    intent = t['intent']
    sentence = t['text']
    entity_map = {}
    for entity in t['entities']:
        new_label_df.append({
            'sentence': sentence,
            'query': intent+'__'+ entity['entity'],
            'answer': entity['value']
        })


# In[ ]:


new_label_df = pd.DataFrame(new_label_df)


# In[ ]:


new_label_df.head()


# In[ ]:


new_label_df['id'] = new_label_df.index


# In[ ]:


new_label_df.head()


# # read original data

# In[ ]:


new_label_train, new_label_eval = train_test_split(new_label_df, test_size=0.3)


# In[ ]:


original_train_df = pd.read_csv("../data/event_type_entity_extract_train.csv", names=['id', 'sentence', 'query', 'answer'])


# In[ ]:


updated_train_df = pd.concat([new_label_df, original_train_df], axis=0)


# In[ ]:


updated_train_df.to_csv("../data/cat_train.csv", index=False, header=False, columns=['id', 'sentence', 'query', 'answer'])


# In[ ]:


new_label_eval.to_csv("../data/cat_eval.csv", index=False, header=False, columns=['id', 'sentence', 'query', 'answer'])


# In[ ]:




