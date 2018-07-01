from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import GRUCell
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


class snli_dec_model(object):
    def __init__(self, voc_size, target_size, input_len_max1, input_len_max2, lr, dev, sess, makedir=True):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(now)
        print("Create snli_dec_model class...")
        print()

        self.voc_size = voc_size
        self.target_size = target_size
        self.input_len_max1 = input_len_max1
        self.input_len_max2 = input_len_max2
        self.lr = lr
        self.sess = sess
        self.dev = dev
        self.makedir = makedir
        
        self._build_graph()
        self.sess.run(tf.global_variables_initializer())
        

    def _build_graph(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(now)
        print("Build Graph...")
        print()
    
        self.xavier_init = tf.contrib.layers.xavier_initializer()
        
        self.embed_dim = 100
        self.state_dim = 100
        self.bi_state_dim = self.state_dim * 2
        self.attend_dim = self.bi_state_dim
        self.fc_dim = 250
        
        print("embed_dim : %d" % self.embed_dim)
        print("state_dim : %d" % self.state_dim)
        print("bi_state_dim : %d" % self.bi_state_dim)
        print("attend_dim : %d" % self.attend_dim)
        print("fc_dim : %d" % self.fc_dim)
        print()
        
        with tf.device(self.dev):
            with tf.variable_scope("input_placeholders"):
                self.enc_input1 = tf.placeholder(tf.int32, shape=[None, self.input_len_max1], name="enc_input1")
                self.enc_seq_len1 = tf.placeholder(tf.int32, shape=[None, ], name="enc_seq_len1")
                self.enc_input2 = tf.placeholder(tf.int32, shape=[None, self.input_len_max2], name="enc_input2")
                self.enc_seq_len2 = tf.placeholder(tf.int32, shape=[None, ], name="enc_seq_len2")
                self.targets = tf.placeholder(tf.int32, shape=[None, ], name="targets")
                self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
                self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            with tf.variable_scope("words_embedding"):
                self.embeddings = tf.get_variable("embeddings", [self.voc_size, self.embed_dim], initializer=self.xavier_init)
                self.embed_in1 = tf.nn.embedding_lookup(self.embeddings, self.enc_input1, name="embed_in1")
                self.embed_in2 = tf.nn.embedding_lookup(self.embeddings, self.enc_input2, name="embed_in2")
                
                self.pad_mask1 = tf.sequence_mask(self.enc_seq_len1, self.input_len_max1, dtype=tf.float32, name="pad_mask1")
                self.pad_mask2 = tf.sequence_mask(self.enc_seq_len2, self.input_len_max2, dtype=tf.float32, name="pad_mask2")

            with tf.variable_scope("projection_layer") as scope_rnn:
                self.h1 = tf.layers.dense(self.embed_in1, self.state_dim, activation=tf.nn.relu, name="words_mlp")
                print("h1.get_shape() : %s" % (self.h1.get_shape()))
                
                self.h2 = tf.layers.dense(self.embed_in2, self.state_dim, activation=tf.nn.relu, name="words_mlp", reuse=True)
                print("h2.get_shape() : %s" % (self.h2.get_shape()))
                
            with tf.variable_scope("attention_layer"):
                self.e = tf.tanh(tf.matmul(self.h1, self.h2, transpose_b=True), name="e")
                print("e.get_shape() : %s" % (self.e.get_shape()))
                self.e_t = tf.transpose(self.e, [0, 2, 1], name="e_t")
                print("e_t.get_shape() : %s" % (self.e_t.get_shape()))

                self.attention1 = tf.nn.softmax(self.e, name="attention1")
                self.pad_mask2_ex = tf.ones_like(self.attention1) * tf.reshape(self.pad_mask2, [-1, 1, self.input_len_max2])
                self.attention1 = self.attention1 * self.pad_mask2_ex
                self.attention1 = self.attention1 / tf.reshape(tf.reduce_sum(self.attention1, axis=2), [-1, self.input_len_max1, 1])
                print("attention1.get_shape() : %s" % (self.attention1.get_shape()))

                self.beta = tf.matmul(self.attention1, self.h2, name="beta")
                print("beta.get_shape() : %s" % (self.beta.get_shape()))

                self.attention2 = tf.nn.softmax(self.e_t, name="attention2")
                self.pad_mask1_ex = tf.ones_like(self.attention2) * tf.reshape(self.pad_mask1, [-1, 1, self.input_len_max1])
                self.attention2 = self.attention2 * self.pad_mask1_ex
                self.attention2 = self.attention2 / tf.reshape(tf.reduce_sum(self.attention2, axis=2), [-1, self.input_len_max2, 1])
                print("attention2.get_shape() : %s" % (self.attention2.get_shape()))

                self.alpha = tf.matmul(self.attention2, self.h1, name="alpha")
                print("alpha.get_shape() : %s" % (self.alpha.get_shape()))
                                                                                    
            with tf.variable_scope("comparison_layer"):
                self.aligned1 = tf.concat([self.h1, self.beta], axis=2, name="aligned1")
                self.aligned2 = tf.concat([self.h2, self.alpha], axis=2, name="aligned2")
                
                self.v1 = tf.layers.dense(self.aligned1, self.state_dim*2, activation=tf.nn.relu, name="g1")
                self.v1 = tf.layers.dense(self.v1, self.state_dim*2, activation=tf.nn.relu, name="g2")
                print("v1.get_shape() : %s" % (self.v1.get_shape()))
                
                self.v2 = tf.layers.dense(self.aligned2, self.state_dim*2, activation=tf.nn.relu, name="g1", reuse=True)
                self.v2 = tf.layers.dense(self.v2, self.state_dim*2, activation=tf.nn.relu, name="g2", reuse=True)
                print("v2.get_shape() : %s" % (self.v2.get_shape()))

            with tf.variable_scope("aggregation_layer"):
                self.v1_agg = tf.reduce_sum(self.v1, axis=1, name="v1_agg")
                print("v1_agg.get_shape() : %s" % (self.v1_agg.get_shape()))

                self.v2_agg = tf.reduce_sum(self.v2, axis=1, name="v2_agg")
                print("v2_agg.get_shape() : %s" % (self.v2_agg.get_shape()))

                self.v_agg = tf.concat([self.v1_agg, self.v2_agg], axis=1, name="v_agg")
                print("v_agg.get_shape() : %s" % (self.v_agg.get_shape()))

            with tf.variable_scope("dense_layer"):
                self.y_hat = tf.layers.dense(self.v_agg, self.fc_dim, activation=tf.tanh, name="out1")
                self.y_hat = tf.layers.dense(self.y_hat, self.target_size, name="out2")
                print("y_hat.get_shape() : %s" % (self.y_hat.get_shape()))
                
            with tf.variable_scope("train_optimization"):
                self.train_vars = tf.trainable_variables()
                
                print()
                print("trainable_variables")
                for varvar in self.train_vars:
                    print(varvar)
                print()
                
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_hat, labels=self.targets)
                self.loss = tf.reduce_mean(self.loss, name="loss")
                self.loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.train_vars if "bias" not in v.name]) * 0.0001
                self.loss = self.loss + self.loss_l2
                
                self.predict = tf.argmax(tf.nn.softmax(self.y_hat), 1)
                self.predict = tf.cast(tf.reshape(self.predict, [self.batch_size, 1]), tf.int32, name="predict")

                self.target_label = tf.cast(tf.reshape(self.targets, [self.batch_size, 1]), tf.int32)
                self.correct = tf.equal(self.predict, self.target_label)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
                
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                self.decay_rate = tf.maximum(0.00007, 
                                             tf.train.exponential_decay(self.lr, self.global_step, 
                                                                        1500, 0.95, staircase=True), 
                                             name="decay_rate")
                self.opt = tf.train.AdamOptimizer(learning_rate=self.decay_rate)
                self.grads_and_vars = self.opt.compute_gradients(self.loss, self.train_vars)
                self.grads_and_vars = [(tf.clip_by_norm(g, 30.0), v) for g, v in self.grads_and_vars]
                self.grads_and_vars = [(tf.add(g, tf.random_normal(tf.shape(g), stddev=0.001)), v) for g, v in self.grads_and_vars]

                self.train_op = self.opt.apply_gradients(self.grads_and_vars, global_step=self.global_step, name="train_op")
            
            if self.makedir == True:
                # Summaries for loss and lr
                self.loss_summary = tf.summary.scalar("loss", self.loss)
                self.accuracy_summary = tf.summary.scalar("accuracy", self.accuracy)
                self.lr_summary = tf.summary.scalar("lr", self.decay_rate)

                # Output directory for models and summaries
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                self.out_dir = os.path.abspath(os.path.join("./model_vanilla", timestamp))
                print("LOGDIR = %s" % self.out_dir)
                print()

                # Train Summaries
                self.train_summary_op = tf.summary.merge([self.loss_summary, self.accuracy_summary, self.lr_summary])
                self.train_summary_dir = os.path.join(self.out_dir, "summary", "train")
                self.train_summary_writer = tf.summary.FileWriter(self.train_summary_dir, self.sess.graph)

                # Test summaries
                self.test_summary_op = tf.summary.merge([self.loss_summary, self.accuracy_summary, self.lr_summary])
                print(self.test_summary_op)
                self.test_summary_dir = os.path.join(self.out_dir, "summary", "test")
                self.test_summary_writer = tf.summary.FileWriter(self.test_summary_dir, self.sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
                self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model-step")
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

            
    def batch_train(self, batchs, data_x1, data_x2, data_y, len_x1, len_x2, writer=False):
        feed_dict = {self.enc_input1: data_x1, 
                     self.enc_seq_len1: len_x1,
                     self.enc_input2: data_x2, 
                     self.enc_seq_len2: len_x2,
                     self.targets: data_y,
                     self.batch_size: batchs,
                     self.keep_prob: 0.75}

        if writer == True:
            results = \
            self.sess.run([self.train_op, self.predict, self.loss, self.accuracy, self.attention1,
                           self.global_step, self.decay_rate, self.train_summary_op], 
                          feed_dict)
            
            ret = [results[1], results[2], results[3], results[4], results[5], results[6]]
        
            self.train_summary_writer.add_summary(results[7], results[5])
        else:
            results = \
            self.sess.run([self.train_op, self.predict, self.loss, self.accuracy, self.attention1,
                           self.global_step, self.decay_rate],
                          feed_dict)
            
            ret = [results[1], results[2], results[3], results[4], results[5], results[6]]
        
        return ret

    
    def batch_test(self, batchs, data_x1, data_x2, data_y, len_x1, len_x2, writer=False):
        feed_dict = {self.enc_input1: data_x1, 
                     self.enc_seq_len1: len_x1,
                     self.enc_input2: data_x2, 
                     self.enc_seq_len2: len_x2,
                     self.targets: data_y,
                     self.batch_size: batchs,
                     self.keep_prob: 1.0}
        
        if writer == True:
            results = \
                self.sess.run([self.predict, self.loss, self.accuracy, self.attention1,
                               self.global_step, self.decay_rate, self.test_summary_op], 
                              feed_dict)

            ret = [results[0], results[1], results[2], results[3], results[4], results[5]]

            self.test_summary_writer.add_summary(results[6], results[4])
        else:
            results = \
                self.sess.run([self.predict, self.loss, self.accuracy, self.attention1,
                               self.global_step, self.decay_rate], 
                              feed_dict)

            ret = [results[0], results[1], results[2], results[3], results[4], results[5]]
            
        return ret
    
    
    def save_model(self):
        current_step = tf.train.global_step(self.sess, self.global_step)
        self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
        
        
    def load_model(self, file_model):
        print("Load model (%s)..." % file_model)
        #file_model = "./model/2017-12-20 11:19/checkpoints/"
        #self.saver.restore(self.sess, tf.train.latest_checkpoint(file_model))
        self.saver.restore(self.sess, file_model)

