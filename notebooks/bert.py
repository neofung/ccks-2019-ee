#!/usr/bin/env python
# coding: utf-8

# > 参考 https://github.com/bojone/bert_in_keras/blob/master/subject_extract.py

# In[ ]:


import json
from tqdm import tqdm
import os, re
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
import hanlp


# In[ ]:


mode = 0
maxlen = 256
learning_rate = 5e-5
min_learning_rate = 1e-5


# In[ ]:


# config_path = '../data/bert/chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '../data/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '../data/bert/chinese_L-12_H-768_A-12/vocab.txt'

config_path = '../data/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../data/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../data/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

# config_path = '../data/bert/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json'
# checkpoint_path = '../data/bert/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt'
# dict_path = '../data/bert/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt'


# In[ ]:


token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


# class OurTokenizer(Tokenizer):
#     def _tokenize(self, text):
#         R = []
#         items = re.split("[\{\}]", text)
#         unused_count = 2
#         for item in items:
#             if item.startswith('DATE') or item.startswith('NUM'):
#                 c = '[unused%d]' % (unused_count)
#                 unused_count += 1
#                 R.append(c)
#             else:
#                 for c in item:
#                     if c in self._token_dict:
#                         R.append(c)
#                     elif self._is_space(c):
#                         R.append('[unused1]') # space类用未经训练的[unused1]表示
#                     else:
#                         R.append('[UNK]') # 剩余的字符是[UNK]
#         return R

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)


# In[ ]:


# tokenizer.tokenize("因 {DATE}公司转增股本，泰怡凱拟通过二级市场集中竞价交易、大宗交易减持股份数量相应调整为合计不超过 {NUM_1} 股，即不超过公司目前总股本的 {NUM_2}。")


# In[ ]:


D = pd.read_csv('../temp/cat_event_extraction_train.csv', encoding='utf-8', header=None)
D = D[D[2] != u'其他']
classes = set(D[2].unique())


# In[ ]:


print(classes)


# In[ ]:


train_data = []
for t,c,n in zip(D[1], D[2], D[3]):
    train_data.append((t, c, n))


if True or not os.path.exists('../temp/bert_random_order_train.json'):
    random_order = list(range(len(train_data)))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('../temp/bert_random_order_train.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('../temp/bert_random_order_train.json'))


# In[ ]:


# train_data


# In[ ]:


dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == mode]
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]
additional_chars = set()
for d in train_data + dev_data:
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', d[2]))

additional_chars.remove('，')


# In[ ]:


additional_chars


# In[ ]:


D = pd.read_csv('../temp/cat_event_extraction_eval.csv', encoding='utf-8', header=None)
test_data = []
for id,t,c in zip(D[0], D[1], D[2]):
    test_data.append((id, t, c))


# In[ ]:


train_data[0]


# In[ ]:


tokenizer.tokenize('勤上光电')


# In[ ]:


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i+n_list2] == list2:
            return i
    return -1


class data_generator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, S1, S2 = [], [], [], []
            for i in idxs:
                d = self.data[i]
                text, c = d[0][:maxlen], d[1]
                text = '___%s___%s' % (c, text)
                tokens = tokenizer.tokenize(text)
                e = d[2]
                e_tokens = tokenizer.tokenize(e)[1:-1]
                s1, s2 = np.zeros(len(tokens)), np.zeros(len(tokens))
                start = list_find(tokens, e_tokens)
                if start != -1:
                    end = start + len(e_tokens) - 1
                else:
                    start = len(e_tokens) -1
                    end = len(e_tokens) - 1
                    
                s1[start] = 1
                s2[end] = 1
                x1, x2 = tokenizer.encode(first=text)
                X1.append(x1)
                X2.append(x2)
                S1.append(s1)
                S2.append(s2)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    S1 = seq_padding(S1)
                    S2 = seq_padding(S2)
                    yield [X1, X2, S1, S2], None
                    X1, X2, S1, S2 = [], [], [], []


# In[ ]:


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


# In[ ]:


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


# In[ ]:


# bert_model = load_trained_model_from_checkpoint(
#     config_file=config_path,
#     checkpoint_file=checkpoint_path,
#     training=False,
#     trainable=True,
#     use_task_embed=True,
#     task_num=10,
#     seq_len=None
# )


bert_model = load_trained_model_from_checkpoint(config_path, 
                                                checkpoint_path, 
                                                seq_len=None)

for l in bert_model.layers:
    l.trainable = True


x1_in = Input(shape=(None,)) # 待识别句子输入
x2_in = Input(shape=(None,)) # 待识别句子输入
s1_in = Input(shape=(None,)) # 实体左边界（标签）
s2_in = Input(shape=(None,)) # 实体右边界（标签）

x1, x2, s1, s2 = x1_in, x2_in, s1_in, s2_in
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)

x = bert_model([x1, x2])
ps1 = Dense(1, use_bias=False)(x)
ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])
ps2 = Dense(1, use_bias=False)(x)
ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])


# In[ ]:


model = Model([x1_in, x2_in], [ps1, ps2])


train_model = Model([x1_in, x2_in, s1_in, s2_in], [ps1, ps2])

loss1 = K.mean(K.categorical_crossentropy(s1_in, ps1, from_logits=True))
ps2 -= (1 - K.cumsum(s1, 1)) * 1e10
loss2 = K.mean(K.categorical_crossentropy(s2_in, ps2, from_logits=True))
loss = loss1 + loss2

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(learning_rate))
train_model.summary()


# In[ ]:


def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


# In[ ]:


def extract_entity(text_in, c_in):
    if c_in not in classes:
        return 'NaN'
    text_in = '___%s___%s' % (c_in, text_in)
    text_in = text_in[:510]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first=text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
    _ps1, _ps2  = model.predict([_x1, _x2])
    _ps1, _ps2 = softmax(_ps1[0]), softmax(_ps2[0])
    for i, _t in enumerate(_tokens):
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            _ps1[i] -= 10
    start = _ps1.argmax()
    for end in range(start, len(_tokens)):
        _t = _tokens[end]
        if len(_t) == 1 and re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', _t) and _t not in additional_chars:
            break
    end = _ps2[start:end+1].argmax() + start
    a = text_in[start-1: end]
#     print(text_in)
#     print(start-1, end)
#     print(text_in[:start-1])
    return a


# In[ ]:


class Evaluate(Callback):
    def __init__(self):
        self.ACC = []
        self.best = 0.
        self.passed = 0
    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
    def on_epoch_end(self, epoch, logs=None):
        acc = self.evaluate()
        self.ACC.append(acc)
        if acc > self.best:
            self.best = acc
            train_model.save_weights('../models/bert_best_model.weights')
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))
    def evaluate(self):
        A = 1e-10
        with open('../temp/dev_pred.json', 'w', encoding='utf-8')as F:
            for d in tqdm(iter(dev_data)):
                R = extract_entity(d[0], d[1])
                if R == d[2]:
                    A += 1
                s = ', '.join(d)
                s += ', '
                s += R
                F.write(s + '\n')
        return A / len(dev_data)


# In[ ]:


def test(test_data):
    """注意官方页面写着是以\t分割，实际上却是以逗号分割
    """
    with open('../temp/result.txt', 'w', encoding='utf-8') as F:
        for d in tqdm(iter(test_data)):
            s = '"%s","%s"\n' % (d[0], extract_entity(d[1].replace('\t', ''), d[2]))
            F.write(s)


# In[ ]:


evaluator = Evaluate()
train_D = data_generator(train_data)


# In[ ]:


train_model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=len(train_D),
                              epochs=100,
                              callbacks=[evaluator]
                             )


# In[ ]:


train_model.load_weights('../models/bert_best_model.weights')


# In[ ]:


extract_entity("因个人原因，杨永平先生申请辞去所担任的董事、总经理、战略与投资管理委员会委员等职务。",
               "辞职__主体")


# In[ ]:


test(test_data)


# In[ ]:


import joblib
intent_entity_dict = joblib.load("../temp/intent_entity_dict.pkl")


# In[ ]:


import re
D = pd.read_csv('../temp/classification_result.txt', encoding='utf-8', header=None)
results = []

processed_ids = set()

with open('../temp/result.txt', 'w', encoding='utf-8') as F:
    for id,sentence,c in tqdm(zip(D[0], D[1], D[2]), total=13633):
        t = re.sub("\d{4}\s*年\d{1,2}\s*月\d{1,2}\s*日", "{DATE}", sentence)
        t_ids= id.split("_")
        id = '_'.join(t_ids[:3])
        processed_ids.add(id)
        if len(processed_ids) >100:
            break
        if c=='其他':
            results.append({
                "id": id,
                "sentence": t,
            })
        else:
            entities = intent_entity_dict[c]
            for entity in entities:
                query = c+"__"+entity
                answer = extract_entity(t.replace('\t', ''), query)
                if answer and answer!='_':
                    results.append({
                        "id": id,
                        "sentence": t,
                        "subject": c,
                        "predicate": entity,
                        "object": answer
                    })


# In[ ]:


results_df = pd.DataFrame(results)
results_df


# In[ ]:


results_df.fillna("", inplace=True)


# In[ ]:


results_df.to_excel("../temp/spo_extraction_results.xlsx",index=False, encoding='utf-8', columns=['id', 'sentence', 'subject', 'predicate', 'object'])


# In[ ]:




