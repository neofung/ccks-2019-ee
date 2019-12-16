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


# In[ ]:


mode = 0
maxlen = 320
learning_rate = 3e-5
min_learning_rate = 3e-6


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


D = pd.read_csv('../temp/cat_classification_train.csv', encoding='utf-8', header=None)
# D = D[D[2] != u'其他']
classes = list(set(D[2].unique()))
classes.sort()


# In[ ]:


classes2id = {}
for i, c in enumerate(classes):
    classes2id[c]=i
classes2id


# In[ ]:


train_data = []
for t,c in zip(D[1], D[2]):
    train_data.append((t, classes2id[c]))


if True or not os.path.exists('../temp/bert_classification_random_order_train.json'):
    random_order = list(range(len(train_data)))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('../temp/bert_classification_random_order_train.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('../temp/bert_classification_random_order_train.json'))


# In[ ]:


# train_data


# In[ ]:


dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == mode]
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]
additional_chars = set()


# In[ ]:


additional_chars


# In[ ]:


D = pd.read_csv('../temp/cat_classification_eval.csv', encoding='utf-8', header=None)
test_data = []
for id,t in zip(D[0], D[1]):
    test_data.append((id, t))


# In[ ]:


train_data[0]


# In[ ]:


tokenizer.tokenize('勤上光电')


# In[ ]:


tokenizer.encode(first='勤上光电, 测试', second='bert')


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
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


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


# In[ ]:


from keras.utils.np_utils import to_categorical

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
p = Dense(len(classes2id), activation='softmax')(x)

# categorical_labels = to_categorical(int_labels, num_classes=None)


# In[ ]:


model = Model([x1_in, x2_in], p)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate), # 用足够小的学习率
    metrics=['sparse_categorical_accuracy'],
)
model.summary()


# In[ ]:


def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)


# In[ ]:


def predict_event(text_in):
    text_in = text_in[:maxlen]
    _tokens = tokenizer.tokenize(text_in)
    _x1, _x2 = tokenizer.encode(first=text_in)
    _x1, _x2 = np.array([_x1]), np.array([_x2])
    _ps  = model.predict([_x1, _x2])
    _ps = softmax(_ps[0])
    _ps = _ps.argmax()
    a = classes[_ps]
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
            model.save_weights('../models/bert_classification_best_model.weights')
        print('acc: %.4f, best acc: %.4f\n' % (acc, self.best))
    def evaluate(self):
        A = 1e-10
        with open('../temp/bert_classification_dev_pred.json', 'w', encoding='utf-8')as F:
            for d in tqdm(iter(dev_data)):
                R = predict_event(d[0])
                if R == classes[d[1]]:
                    A += 1
                s = ', '.join([str(t) for t in d])
                s += ', '
                s += R
                F.write(s + '\n')
        return A / len(dev_data)


# In[ ]:


def test(test_data):
    """注意官方页面写着是以\t分割，实际上却是以逗号分割
    """
    with open('../temp/classification_result.txt', 'w', encoding='utf-8') as F:
        for d in tqdm(iter(test_data)):
            s = '"%s","%s","%s"\n' % (d[0], d[1], predict_event(d[1].replace('\t', '')))
            F.write(s)


# In[ ]:


evaluator = Evaluate()
train_D = data_generator(train_data)
dev_D = data_generator(dev_data)


# In[ ]:


model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=100,
    validation_data=dev_D.__iter__(),
    validation_steps=len(dev_D),
    callbacks=[evaluator]
)


# In[ ]:


# model.save_weights("../models/bert_classification_best_model.weights")


# In[ ]:


model.load_weights('../models/bert_classification_best_model.weights')


# In[ ]:


predict_event("浙江尤夫高新纤维股份有限公司（以下简称“公司”）董事会于 2017 年 1月 4 日收到公司独立董事王华平先生提交的书面辞职报告，王华平先生因个人原因辞去公司第三届董事会独立董事、公司第三届董事会战略决策委员会以及薪酬与考核委员会职务。")


# In[ ]:


test(test_data)


# In[ ]:


D = pd.read_csv('../temp/sentences_test.tsv', encoding='utf-8', header=None, sep='\t')
test_data = []
for id,t in zip(D[0], D[1]):
    test_data.append((id, t))


# In[ ]:




