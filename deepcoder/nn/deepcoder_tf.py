import collections
import json
import numpy as np
import re

from deepcoder.dsl import impl
from deepcoder.dsl import constants
from deepcoder.dsl.types import INT, LIST
from deepcoder.dsl.value import NULLVALUE
from deepcoder import util

import tensorflow as tf
# K = 256  # number of hidden units
# M = 5   # number of input-output pairs per program
# E = 2   # embedding dimension
# padding length of input
# I = 3   # max number of inputs
L = 20  # length of input


def encode(value, L=L):
    if value.type == LIST:
        typ = [0, 1]
        vals = value.val + [constants.NULL - constants.INTMAX] * (L - len(value.val))
    elif value.type == INT:
        typ = [1, 0]
        vals = [value.val] + [constants.NULL - constants.INTMAX] * (L - 1)
    elif value == NULLVALUE:
        typ = [0, 0]
        vals = [constants.NULL - constants.INTMAX] * L
    return np.array(typ), np.array(vals)

def get_row(examples, max_nb_inputs, L=L):
    row_type = np.zeros((len(examples), max_nb_inputs+1, 2))
    row_val = np.zeros((len(examples), max_nb_inputs+1, L))
    for i, (inputs, output) in enumerate(examples):
        # one problem [[inputs], output]
        for j, input in enumerate(inputs):
            typ, vals = encode(input, L)
            row_type[i][j] = typ
            row_val[i][j] = vals

        for j in range(len(inputs), max_nb_inputs):
            # pad with null
            typ, vals = encode(NULLVALUE, L)
            row_type[i][j] = typ
            row_val[i][j] = vals

        typ, vals = encode(output, L)
        row_type[i][-1] = typ
        row_val[i][-1] = vals
    return row_type, row_val


def get_XY(problems, max_nb_inputs):
    y = []
    rows_type = []
    rows_val = []
    # print(problems)
    for problem in problems:
        examples = [util.decode_example(x) for x in problem['examples']]
        # print(examples[0])
        row_type, row_val = get_row(examples, max_nb_inputs, L)
        # print(row)
        y.append(problem['attribute'])
        rows_type.append(row_type)
        rows_val.append(row_val)

    y = np.array(y)
    rows_type = np.array(rows_type)
    rows_val = np.array(rows_val)

    # preprocess
    rows_val += np.ones_like(rows_val) * constants.INTMAX

    # print(1111, X)
    return rows_type, rows_val, y


class Deepcoder:
    def __init__(self, I, E, K=256, lr=1e-3):
        self.I = I
        self.E = E
        self.dim = K

        self.lr = lr

        self.sess = tf.Session()

        self.get_model(self.I, self.E)

        self.writer = tf.summary.FileWriter("../logs/", self.sess.graph)
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

    def get_model(self, I, E, M=5):
        """
        Arguments:
            I (int): number of inputs in each program. input count is
                padded to I with null type and vals.
            E (int): embedding dimension
            M (int): number of examples per program. default 5.
        """
        self.type_ph = tf.placeholder(tf.float32, [None, M, I + 1, 2], name='type')
        self.val_ph = tf.placeholder(tf.int32, [None, M, I + 1, L], name='value')
        self.label_ph = tf.placeholder(tf.float32, [None, len(impl.FUNCTIONS)], name='labels')

        number_embeddings = tf.get_variable('number_embeddings', [constants.NULL + 1, self.E])
        embedded_vals = tf.nn.embedding_lookup(number_embeddings, self.val_ph)

        reshaped_vals = tf.reshape(embedded_vals, [-1, M, I + 1, L * E])

        concated = tf.concat([self.type_ph, reshaped_vals], axis=-1)

        flattened = tf.reshape(concated, [-1, M, (I + 1) * (L * E + 2)])
        x1 = tf.layers.dense(flattened, self.dim, activation=tf.nn.sigmoid)
        x2 = tf.layers.dense(x1, self.dim, activation=tf.nn.sigmoid)
        x3 = tf.layers.dense(x2, self.dim, activation=tf.nn.sigmoid)

        ave = tf.reduce_mean(x3, axis=1)
        pred = tf.layers.dense(ave, len(impl.FUNCTIONS), activation=None)
        self.pred = tf.nn.sigmoid(pred)

        with tf.name_scope('train_loss'):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_ph, logits=pred))

            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.merged = tf.summary.merge_all()

    def fit(self, rows_type, rows_val, y, epochs, validation_split):
        for i in range(epochs):
            print('start epochs', i)
            _, summary, loss = self.sess.run([self.train_op, self.merged, self.loss], feed_dict={self.type_ph: rows_type,
                                                                                self.val_ph: rows_val,
                                                                                self.label_ph: y})
            self.writer.add_summary(summary, i)
            print('loss:', loss)

    def save(self, outfile="models/deepcoder.ckpt"):
        self.saver.save(self.sess, outfile)

    def load(self, outfile="models/deepcoder.ckpt"):
        ckpt = tf.train.get_checkpoint_state("models/train/")
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def predict(self, rows_type, rows_val):
        pred = self.sess.run(self.pred, feed_dict={self.type_ph: rows_type, self.val_ph: rows_val})
        return pred
