from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import collections
import os
import json
import re
import collections
import datetime
import pickle
import random
from tqdm import trange
import time

class snli_data_batcher(object):
    def __init__(self, data_idx_list_train, data_idx_list_dev, data_idx_list_test, voc_size, dic, seq_max1, seq_max2, cat_cnt):
        self.data_idx_list_train = data_idx_list_train
        self.data_idx_list_dev = data_idx_list_dev
        self.data_idx_list_test = data_idx_list_test
        self.dic = dic
        self.voc_size = voc_size
        self.seq_max1 = seq_max1
        self.seq_max2 = seq_max2
        self.cat_cnt = cat_cnt
        
        self.train_size = len(self.data_idx_list_train)
        self.dev_size = len(self.data_idx_list_dev)
        self.test_size = len(self.data_idx_list_test)


    def get_rand_batch(self, size, data="train"):
        np.random.seed(seed=int(time.time()))

        data_x1 = np.zeros((size, self.seq_max1), dtype=np.int)
        data_x2 = np.zeros((size, self.seq_max2), dtype=np.int)
        data_y = np.zeros(size, dtype=np.int)
        len_x1 = np.zeros(size, dtype=np.int)
        len_x2 = np.zeros(size, dtype=np.int)

        if data == "dev":
            assert size <= len(self.data_idx_list_dev)
            index = np.random.choice(range(len(self.data_idx_list_dev)), size, replace=False)

        elif data == "test":
            assert size <= len(self.data_idx_list_test)
            index = np.random.choice(range(len(self.data_idx_list_test)), size, replace=False)

        else:
            assert size <= len(self.data_idx_list_train)
            index = np.random.choice(range(len(self.data_idx_list_train)), size, replace=False)

        for a in range(len(index)):
            idx = index[a]

            if data == "dev":
                s1 = self.data_idx_list_dev[idx][2]
                s2 = self.data_idx_list_dev[idx][3]
                label = self.data_idx_list_dev[idx][0]
            elif data == "test":
                s1 = self.data_idx_list_test[idx][2]
                s2 = self.data_idx_list_test[idx][3]
                label = self.data_idx_list_test[idx][0]
            else:
                s1 = self.data_idx_list_train[idx][2]
                s2 = self.data_idx_list_train[idx][3]
                label = self.data_idx_list_train[idx][0]

            x1 = s1 + [self.dic["<pad>"]] * (self.seq_max1 - len(s1))
            x2 = s2 + [self.dic["<pad>"]] * (self.seq_max2 - len(s2))
            y = label

            assert len(x1) == self.seq_max1
            assert max(x1) < self.voc_size
            assert len(x2) == self.seq_max2
            assert max(x2) < self.voc_size
            assert y < self.cat_cnt

            data_x1[a] = x1
            data_x2[a] = x2
            data_y[a] = y
            len_x1[a] = len(s1)
            len_x2[a] = len(s2)

        return data_x1, data_x2, data_y, len_x1, len_x2


    def get_step_batch(self, start, size, data="train"):
        np.random.seed(seed=int(time.time()))

        data_x1 = np.zeros((size, self.seq_max1), dtype=np.int)
        data_x2 = np.zeros((size, self.seq_max2), dtype=np.int)
        data_y = np.zeros(size, dtype=np.int)
        len_x1 = np.zeros(size, dtype=np.int)
        len_x2 = np.zeros(size, dtype=np.int)

        if data == "dev":
            assert start+size <= len(self.data_idx_list_dev)
        elif data == "test":
            assert start+size <= len(self.data_idx_list_test)
        else:
            assert start+size <= len(self.data_idx_list_train)

        for a in range(size):

            if data == "dev":
                s1 = self.data_idx_list_dev[start+a][2]
                s2 = self.data_idx_list_dev[start+a][3]
                label = self.data_idx_list_dev[start+a][0]
            elif data == "test":
                s1 = self.data_idx_list_test[start+a][2]
                s2 = self.data_idx_list_test[start+a][3]
                label = self.data_idx_list_test[start+a][0]
            else:
                s1 = self.data_idx_list_train[start+a][2]
                s2 = self.data_idx_list_train[start+a][3]
                label = self.data_idx_list_train[start+a][0]

            x1 = s1 + [self.dic["<pad>"]] * (self.seq_max1 - len(s1))
            x2 = s2 + [self.dic["<pad>"]] * (self.seq_max2 - len(s2))
            y = label

            assert len(x1) == self.seq_max1
            assert max(x1) < self.voc_size
            assert len(x2) == self.seq_max2
            assert max(x2) < self.voc_size
            assert y < self.cat_cnt

            data_x1[a] = x1
            data_x2[a] = x2
            data_y[a] = y
            len_x1[a] = len(s1)
            len_x2[a] = len(s2)
        
        #print("%d %d %d %d %d" % (np.max(data_x1), np.max(data_x2), np.max(data_y), np.max(len_x1), np.max(len_x2)))

        return data_x1, data_x2, data_y, len_x1, len_x2