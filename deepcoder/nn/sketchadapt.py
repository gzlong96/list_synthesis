import collections
import json
import numpy as np
import re
import random

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


def encode_program(prefix):
    splited = prefix.split('|')
    input_type = []
    functions = []
    args = np.zeros((27,))
    for statement in splited:
        if statement=='LIST' or statement=='INT':
            input_type.append(statement)
        else:
            splited_stmt = statement.split(',')
            f = impl.NAME2FUNC[splited_stmt[0]]
            functions.append(impl.FUNCTIONS.index(f))
            for arg in splited_stmt[1:]:
                if arg in impl.NAME2FUNC.keys():
                    args[impl.LAMBDAS.index(impl.NAME2FUNC[arg])] = 1
                else:
                    args[19 + int(arg)] = 1

    # return input_type, functions
    return [functions, args]


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


def get_XY(problems, max_nb_inputs, max_nb_tokens):
    y = [[], []]
    rows_type = []
    rows_val = []
    # print(problems)
    for problem in problems:
        examples = [util.decode_example(x) for x in problem['examples']]
        # print(examples[0])
        row_type, row_val = get_row(examples, max_nb_inputs, L)
        # print(row)
        next_y = encode_program(problem['program'])
        y[0].append(next_y[0])
        y[1].append(next_y[1])
        rows_type.append(row_type)
        rows_val.append(row_val)

    rows_type = np.array(rows_type)
    rows_val = np.array(rows_val)

    # preprocess
    rows_val += np.ones_like(rows_val) * constants.INTMAX

    # print(1111, X)
    return rows_type, rows_val, y  #sketchy = [index]  argy = [11100]


class Sketchadapt:
    def __init__(self, I, E, T=2, K=256, lr=1e-3, batch_size=-1):
        self.I = I
        self.E = E
        self.dim = K
        self.T = int(T)

        self.lr = lr
        self.batch_size = batch_size

        self.sess = tf.Session()

        self.get_generator(self.I, self.E)
        self.get_recognizer(self.I, self.E)

        self.writer = tf.summary.FileWriter("../logs/", self.sess.graph)
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

    def get_generator(self, I, E, M=5):
        """
        Arguments:
            I (int): number of inputs in each program. input count is
                padded to I with null type and vals.
            E (int): embedding dimension
            M (int): number of examples per program. default 5.
        """
        with tf.variable_scope('generator'):
            self.type_ph = tf.placeholder(tf.float32, [None, M, I + 1, 2], name='type')
            self.val_ph = tf.placeholder(tf.int32, [None, M, I + 1, L], name='value')
            self.sketch_ph = tf.placeholder(tf.int32, [None, self.T], name='sketches')

            number_embeddings = tf.get_variable('number_embeddings', [constants.NULL + 1, self.E])
            embedded_vals = tf.nn.embedding_lookup(number_embeddings, self.val_ph)

            reshaped_vals = tf.reshape(embedded_vals, [-1, (I + 1) * L, E]) # [b*M, (I + 1) * L, E]

            with tf.name_scope('io_rnn'):
                io_cell = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell1')

                x, s = tf.nn.dynamic_rnn(io_cell, reshaped_vals, dtype=tf.float32)

                c, h = s

                max_c = tf.reshape(tf.layers.max_pooling1d(tf.reshape(c, [-1, M, self.dim]), M, 1), [-1, self.dim])
                max_h = tf.reshape(tf.layers.max_pooling1d(tf.reshape(h, [-1, M, self.dim]), M, 1), [-1, self.dim])

                self.X_h = tf.nn.rnn_cell.LSTMStateTuple(c=max_c,h=max_h)

            with tf.name_scope('program_rnn'):
                program_cell = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell2')

                x_p, h_p = tf.nn.dynamic_rnn(program_cell, tf.tile(x[:,-1:,:],[1,self.T,1]), dtype=tf.float32) # x_p [b*M, L2, dim]

            x1 = tf.layers.dense(x_p, self.dim, activation=None)
            x2 = tf.reshape(x1, [-1, M, self.T, self.dim])
            pooled = tf.layers.max_pooling2d(x2, [M, 1], [1,1])

            reshaped = tf.reshape(pooled, [-1, self.T, self.dim])

            pred = tf.layers.dense(reshaped, 15)

            self.sketch_pred = tf.nn.softmax(pred, axis=-1)

        with tf.name_scope('train_loss_g'):
            self.g_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.sketch_ph, logits=pred))

            tf.summary.scalar('g_loss', self.g_loss)

        with tf.name_scope('train_g'):
            g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
            self.train_op_g = tf.train.AdamOptimizer(self.lr).minimize(self.g_loss, var_list=g_vars)

        self.merged_g = tf.summary.merge_all()


    def get_recognizer(self, I, E, M=5):
        with tf.variable_scope('recognizer'):
            self.selector = tf.placeholder(tf.float32, [None, 27], name='selector')

            program_embeddings = tf.get_variable('sketch_embeddings', [15, self.dim])
            embedded_vals = tf.nn.embedding_lookup(program_embeddings, self.sketch_ph)

            with tf.name_scope('recognizer_rnn'):
                recognizer_cell = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell3')

                x, h = tf.nn.dynamic_rnn(recognizer_cell, embedded_vals, initial_state=self.X_h)

                ct, ht = h

            x1 = tf.layers.dense(ht, self.dim, activation='relu')

            pred = tf.layers.dense(x1, 27)

            self.r_pred = tf.nn.softmax(pred, axis=-1)

        with tf.name_scope('train_loss_r'):
            # self.r_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph, logits=pred))

            self.r_loss = -tf.reduce_mean(tf.reduce_prod(self.r_pred*self.selector+tf.ones_like(self.selector)-self.selector,axis=-1))

            tf.summary.scalar('r_loss', self.r_loss)

        with tf.name_scope('train_r'):
            r_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='recognizer')
            self.train_op_r = tf.train.AdamOptimizer(self.lr).minimize(self.r_loss, var_list=r_vars)

        self.merged_r = tf.summary.merge_all()


    def fit(self, rows_type, rows_val, y, epochs, validation_split):
        sketch_y, arg_y = y
        for i in range(epochs):
            self.fit_r(rows_type, rows_val, sketch_y, arg_y, 1)
            self.fit_g(rows_type, rows_val, sketch_y, 1)


    def fit_r(self, rows_type, rows_val, sketch_y, arg_y, epochs, validation_split=None):
        for i in range(epochs):
            if self.batch_size==-1:
                print('start epochs', i)
                _, summary, loss = self.sess.run([self.train_op_r, self.merged_r, self.r_loss], feed_dict={self.type_ph: rows_type,
                                                                                                       self.val_ph: rows_val,
                                                                                                       self.sketch_ph: sketch_y,
                                                                                                       self.selector: arg_y})
                self.writer.add_summary(summary, i)
                print('r_loss:', loss)
            else:
                print('start epochs', i)
                # zipped = zip(rows_type, rows_val, y)
                # print(*zipped)
                # random.shuffle(zipped)
                # rows_type, rows_val, y = zipped
                for j in range(0, len(arg_y), self.batch_size):
                    _, summary, loss = self.sess.run([self.train_op_r, self.merged_r, self.r_loss],
                                                     feed_dict={self.type_ph: rows_type[j:j+self.batch_size],
                                                                self.val_ph: rows_val[j:j+self.batch_size],
                                                                self.sketch_ph: sketch_y[j:j+self.batch_size],
                                                                self.selector: arg_y[j:j+self.batch_size]})
                self.writer.add_summary(summary, i)
                print('r_loss:', loss)

    def fit_g(self, rows_type, rows_val, sketch_y, epochs, validation_split=None):
        for i in range(epochs):
            if self.batch_size==-1:
                print('start epochs', i)
                _, summary, loss = self.sess.run([self.train_op_g, self.merged_g, self.g_loss], feed_dict={self.type_ph: rows_type,
                                                                                    self.val_ph: rows_val,
                                                                                    self.sketch_ph: sketch_y})
                self.writer.add_summary(summary, i)
                print('g_loss:', loss)
            else:
                print('start epochs', i)
                # zipped = zip(rows_type, rows_val, y)
                # print(*zipped)
                # random.shuffle(zipped)
                # rows_type, rows_val, y = zipped
                for j in range(0, len(sketch_y), self.batch_size):
                    _, summary, loss = self.sess.run([self.train_op_g, self.merged_g, self.g_loss],
                                                     feed_dict={self.type_ph: rows_type[j:j+self.batch_size],
                                                                self.val_ph: rows_val[j:j+self.batch_size],
                                                                self.sketch_ph: sketch_y[j:j+self.batch_size]})
                self.writer.add_summary(summary, i)
                print('g_loss:', loss)


    def save(self, outfile="../models/sketchadapt/model.ckpt"):
        self.saver.save(self.sess, outfile)

    def load(self, outfile="models/deepcoder.ckpt"):
        ckpt = tf.train.get_checkpoint_state("../models/sketchadapt/")
        if ckpt and ckpt.model_checkpoint_path:
            print('load successful')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def predict(self, rows_type, rows_val):
        pred = self.sess.run(self.pred, feed_dict={self.type_ph: rows_type, self.val_ph: rows_val})
        return pred
