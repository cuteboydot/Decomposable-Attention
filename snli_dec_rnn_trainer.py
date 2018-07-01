from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import datetime
import pickle
import random
import logging

from snli_batcher import snli_data_batcher
from snli_dec_rnn_model import snli_dec_model

logging.getLogger("tensorflow").setLevel(logging.WARNING)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(now)
print("Load vocabulary from model file...")

with open("./data/rdic.pickle", 'rb') as handle:
    rdic = pickle.load(handle)
with open("./data/dic.pickle", 'rb') as handle:
    dic = pickle.load(handle)
with open("./data/data_list_idx_train.pickle", 'rb') as handle:
    data_list_idx_train = pickle.load(handle)
with open("./data/data_list_idx_dev.pickle", 'rb') as handle:
    data_list_idx_dev = pickle.load(handle)
with open("./data/data_list_idx_test.pickle", 'rb') as handle:
    data_list_idx_test = pickle.load(handle)
with open("./data/len_max.pickle", 'rb') as handle:
    len_max = pickle.load(handle)

random.shuffle(data_list_idx_train)
random.shuffle(data_list_idx_dev)
random.shuffle(data_list_idx_test)

SIZE_VOC = len(rdic)
print("voc_size = %d" % SIZE_VOC)

SIZE_SENTENCE_MAX1 = len_max[0]
SIZE_SENTENCE_MAX2 = len_max[1]
print("max_sentence_len1 = %d" % SIZE_SENTENCE_MAX1)
print("max_sentence_len2 = %d" % SIZE_SENTENCE_MAX2)
print()

SIZE_TRAIN_DATA = len(data_list_idx_train)
SIZE_DEV_DATA = len(data_list_idx_dev)
SIZE_TEST_DATA = len(data_list_idx_test)
print("dataset for train = %d" % SIZE_TRAIN_DATA)
print("dataset for dev = %d" % SIZE_DEV_DATA)
print("dataset for test = %d" % SIZE_TEST_DATA)
print()

SIZE_TARGET = 3

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print(now)
print("Train start!!")
print()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config) as sess:
    batcher = snli_data_batcher(data_idx_list_train= data_list_idx_train,
                                data_idx_list_dev= data_list_idx_dev,
                                data_idx_list_test= data_list_idx_test,
                                voc_size= SIZE_VOC,
                                dic= dic,
                                seq_max1= SIZE_SENTENCE_MAX1,
                                seq_max2= SIZE_SENTENCE_MAX2,
                                cat_cnt= SIZE_TARGET)

    model = snli_dec_model(voc_size= SIZE_VOC,
                           target_size= SIZE_TARGET,
                           input_len_max1= SIZE_SENTENCE_MAX1,
                           input_len_max2= SIZE_SENTENCE_MAX2,
                           lr= 0.001,
                           dev= "/cpu:0",
                           sess= sess,
                           makedir= True)

    BATCHS = 400
    BATCHS_TEST = len(data_list_idx_test)
    EPOCHS = 6
    STEPS = int(len(data_list_idx_train) / BATCHS) + 1
    loop_step = 0

    for epoch in range(EPOCHS):
        for step in range(STEPS):
            if step == STEPS-1:
                batchs = len(data_list_idx_train) % BATCHS
            else:
                batchs = BATCHS

            writer = False
            if loop_step % 50 == 0:
                writer = True

            data_x1, data_x2, data_y, len_x1, len_x2 = batcher.get_step_batch(start=step*BATCHS, size=batchs, data="train")

            results = model.batch_train(batchs, data_x1, data_x2, data_y, len_x1, len_x2, writer)
            batch_pred = results[0]
            batch_loss = results[1]
            batch_acc = results[2]
            batch_att = results[3]
            g_step = results[4]
            batch_lr = results[5]

            if loop_step % 200 == 0:
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("epoch[%03d] glob_step[%06d] - batch_loss:%.4f, batch_acc:%.4f, lr=%.6f  (%s)" %
                      (epoch, g_step, batch_loss, batch_acc, batch_lr, now))

            if loop_step % 1000 == 0:
                data_x1, data_x2, data_y, len_x1, len_x2 = batcher.get_step_batch(start=0, size=BATCHS_TEST, data="test")

                results = model.batch_test(BATCHS_TEST, data_x1, data_x2, data_y, len_x1, len_x2, writer)
                batch_pred = results[0]
                batch_loss = results[1]
                batch_acc = results[2]
                batch_att = results[3]
                g_step = results[4]
                batch_lr = results[5]

                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("epoch[%03d] glob_step[%06d] - test_loss: %.4f, test_acc: %.4f, lr=%.6f  (%s)" %
                      (epoch, g_step, batch_loss, batch_acc, batch_lr, now))

            loop_step += 1

        if epoch % 2 == 0:
            model.save_model()

    # visualize..
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    for a in range(30):
        test_id = 2*a + 100
        sentence1 = [rdic[w] for w in data_x1[test_id] if w != 0]
        sentence2 = [rdic[w] for w in data_x2[test_id] if w != 0]
        len1 = len_x1[test_id]
        len2 = len_x2[test_id]
        attend = batch_att[test_id]

        print("sentence1:%s" % sentence1)
        print("sentence2:%s" % sentence2)
        print("target:%d, Predict:%d" % (data_y[test_id], batch_pred[test_id]))

        plt.clf()
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111)
        im = ax.imshow(attend[:len1, :len2], cmap="YlOrBr")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)

        ax.set_xticks(range(len2))
        ax.set_xticklabels(sentence2, fontsize=14, rotation=90)
        ax.set_yticks(range(len1))
        ax.set_yticklabels(sentence1, fontsize=14, rotation=0)

        ax.grid()
        plt.show()
        print("~" * 50)


    print("train end...")