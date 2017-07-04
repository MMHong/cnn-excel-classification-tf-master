# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:20:54 2017

@author: ASUS
"""


import itertools
from collections import Counter
import numpy as np
import re
import csv
#import ci
#单行最大字词长度
max_len = 200
#利用正则处理csv文本，去掉一切标点
def clean_str(string):
#    string = unicode(string, 'utf-8')
    string = re.sub("[0-9a-zA-Z～@+*\s]", "", string)
    string = re.sub("[：；，、×.“”…_【】（）():,\#《》〈丶&\-\[\]\/]", "", string)
    string = re.sub("[。！？]", "", string)
    return string[0: max_len]
#CSV文件所在的目，用\转义
path = "E:\\项目1\\cnn-text-classification-tf-master\\traindata.csv"
#读取文件内容为UTF-8编码
with open(path, 'r',encoding='utf-8') as csvfile:
    spamreader = csv.reader(csvfile)
    texts = [[clean_str(row[1]), row[2]] for row in spamreader]
    sentences = [text[0] for text in texts]
#获得字词
def get_vocabulary():
    char_dict = dict()
    for sentence in sentences:
 #     a=ci.cut(sentence)
        for cn_char in sentence:
            if cn_char not in char_dict:
                char_dict[cn_char] = len(char_dict) + 1
    return char_dict
#获得分类列表
def get_labels_list():
    classifies_set = set()
    for text in texts:
        classifies_set.add(text[1])
    return list(classifies_set)
#下载数据和分类
def load_data_and_labels():
    sentences_digital = []
    char_dict = get_vocabulary()
    for sentence in sentences:
        sentence_digital = np.zeros(max_len)
        for i, cn_char in enumerate(sentence):
            sentence_digital[i] = char_dict[cn_char]
        sentences_digital.append(sentence_digital)

    classifies_list = get_labels_list()
    classifies_size = len(classifies_list)
    labels_dicts = dict()
    for i, classify in enumerate(classifies_list):
        classify_label = np.zeros(classifies_size)
        classify_label[i] = 1
        labels_dicts[classify] = classify_label
    for label in labels_dicts:
        print (label, labels_dicts[label])
    labels = [labels_dicts[text[1]] for text in texts]
    return [np.array(sentences_digital), np.array(labels)]

#批处理迭代器
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
