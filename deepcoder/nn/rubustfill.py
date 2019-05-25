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
        typ = [[0, 1] for _ in range(len(value.val))] + [[0, 0] for _ in range(L - len(value.val))]
        vals = value.val + [constants.NULL] * (L - len(value.val))
    elif value.type == INT:
        typ = [[1, 0]] + [[0, 0] for _ in range(L - 1)]
        vals = [value.val] + [constants.NULL] * (L - 1)
    elif value == NULLVALUE:
        typ = [[0, 0] for _ in range(L)]
        vals = [constants.NULL] * L
    return np.array(typ), np.array(vals)

def encode_program(f, max_nb_tokens):
    for i in range(len(f), max_nb_tokens):
        f.append(len(impl.FUNCTIONS)+7)
    return f

def get_row(examples, max_nb_inputs, L=L):
    row_type = np.zeros((len(examples), max_nb_inputs+1, L, 2))
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
    y = []
    rows_type = []
    rows_val = []
    # print(problems)
    for problem in problems:
        examples = [util.decode_example(x) for x in problem['examples']]
        # print(examples[0])
        row_type, row_val = get_row(examples, max_nb_inputs, L)
        # print(row)
        f_list = util.get_program_vec(problem['program'])
        y.append(encode_program(f_list, max_nb_tokens))
        rows_type.append(row_type)
        rows_val.append(row_val)

    y = np.array(y)
    rows_type = np.array(rows_type)
    rows_val = np.array(rows_val)

    # preprocess
    rows_val += np.ones_like(rows_val) * constants.INTMAX

    # print(1111, X)
    # print(rows_val)
    # print(y)
    return rows_type, rows_val, y


class Rubustfill:
    def __init__(self, I, E, L2, K=256, lr=1e-3, batch_size=-1, attention='A'):
        self.I = I
        self.E = E
        self.L2 = L2  # max length of tokens
        self.dim = K

        self.lr = lr
        self.batch_size = batch_size
        self.attention = attention

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
        self.type_ph = tf.placeholder(tf.float32, [None, M, I + 1, L, 2], name='type')
        self.val_ph = tf.placeholder(tf.int32, [None, M, I + 1, L], name='value')
        self.label_ph = tf.placeholder(tf.int32, [None, self.L2], name='labels')

        number_embeddings = tf.get_variable('number_embeddings', [constants.NULL + constants.INTMAX + 1, self.E])
        embedded_vals = tf.nn.embedding_lookup(number_embeddings, self.val_ph)

        concated = tf.concat([self.type_ph, embedded_vals], axis=-1)  # [b, M, I+1, L, E+2]

        reshaped_i_vals = tf.reshape(concated[:,:,:I,:,:], [-1, I * L, E + 2]) # [b*M, I * L, E + 2]
        reshaped_o_vals = tf.reshape(concated[:,:,I:,:,:], [-1, L, E + 2])  # [b*M, L, E + 2]

        if self.attention is None:
            with tf.name_scope('i_rnn'):
                i_cell = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell1')

                x1, h1 = tf.nn.dynamic_rnn(i_cell, reshaped_i_vals, dtype=tf.float32)

            with tf.name_scope('o_rnn'):
                o_cell = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell2')

                x2, h2 = tf.nn.dynamic_rnn(o_cell, reshaped_o_vals, dtype=tf.float32, initial_state=h1)

            with tf.name_scope('program_rnn'):
                program_cell = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell3')

                x_p, h_p = tf.nn.dynamic_rnn(program_cell, tf.tile(x2[:,-1:,:],[1,self.L2,1]), dtype=tf.float32) # x_p [b*M, L2, dim]
        elif self.attention is 'bi':
            with tf.name_scope('i_rnn'):
                i_cell1 = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell11')
                i_cell2 = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell12')

                x11, h11 = tf.nn.dynamic_rnn(i_cell1, reshaped_i_vals, dtype=tf.float32)
                x12, h12 = tf.nn.dynamic_rnn(i_cell2, tf.reverse(reshaped_i_vals,axis=[1]), dtype=tf.float32)

            with tf.name_scope('o_rnn'):
                o_cell1 = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell21')
                o_cell2 = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell22')

                x21, h21 = tf.nn.dynamic_rnn(o_cell1, reshaped_o_vals, dtype=tf.float32, initial_state=h11)
                x22, h22 = tf.nn.dynamic_rnn(o_cell2, tf.reverse(reshaped_o_vals,axis=[1]), dtype=tf.float32, initial_state=h12)

            with tf.name_scope('program_rnn'):
                program_cell1 = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell31')
                program_cell2 = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell32')

                x_p1, h_p1 = tf.nn.dynamic_rnn(program_cell1, tf.tile(x21[:,-1:,:],[1,self.L2,1]), dtype=tf.float32) # x_p [b*M, L2, dim]
                x_p2, h_p2 = tf.nn.dynamic_rnn(program_cell1, tf.tile(x22[:, -1:, :], [1, self.L2, 1]),
                                             dtype=tf.float32)  # x_p [b*M, L2, dim]
                x_p = tf.concat([x_p1, x_p2], axis=-1)

        elif self.attention == 'A':
            with tf.name_scope('i_rnn'):
                i_cell = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell1')

                x1, h1 = tf.nn.dynamic_rnn(i_cell, reshaped_i_vals, dtype=tf.float32)

            with tf.name_scope('o_rnn'):
                o_cell = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell2')

                x2, h2 = self.attention_rnn(o_cell, reshaped_o_vals, x1 ,h1, I*L, L)

            with tf.name_scope('program_rnn'):
                program_cell = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell3')

                x_p, h_p = self.attention_rnn(program_cell, tf.tile(x2[:, -1:, :], [1, self.L2, 1]), x2, h2, L, self.L2)

        elif self.attention == 'B':
            with tf.name_scope('i_rnn'):
                i_cell = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell1')

                x1, h1 = tf.nn.dynamic_rnn(i_cell, reshaped_i_vals, dtype=tf.float32)

            with tf.name_scope('o_rnn'):
                o_cell = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell2')

                x2, h2 = self.attention_rnn(o_cell, reshaped_o_vals, x1 ,h1, I*L, L)

            with tf.name_scope('program_rnn'):
                program_cell = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell3')

                x_p, h_p = self.double_attention_rnn(program_cell, tf.tile(x2[:, -1:, :], [1, self.L2, 1]), x1, x2, h2, I*L, L, self.L2)

        elif self.attention is 'C':
            with tf.name_scope('i_rnn'):
                i_cell1 = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell11')
                i_cell2 = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell12')

                x11, h11 = tf.nn.dynamic_rnn(i_cell1, reshaped_i_vals, dtype=tf.float32)
                x12, h12 = tf.nn.dynamic_rnn(i_cell2, tf.reverse(reshaped_i_vals,axis=[1]), dtype=tf.float32)

            with tf.name_scope('o_rnn'):
                o_cell1 = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell21')
                o_cell2 = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell22')

                x21, h21 = self.attention_rnn(o_cell1, reshaped_o_vals, x11, h11, I * L, L)
                x22, h22 = self.attention_rnn(o_cell2, tf.reverse(reshaped_o_vals,axis=[1]), x12, h12, I * L, L)


            with tf.name_scope('program_rnn'):
                program_cell1 = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell31')
                program_cell2 = tf.nn.rnn_cell.LSTMCell(self.dim, name='cell32')

                x_p1, h_p1 = self.double_attention_rnn(program_cell1, tf.tile(x21[:, -1:, :], [1, self.L2, 1]), x11, x21, h21,
                                                     I * L, L, self.L2)
                x_p2, h_p2 = self.double_attention_rnn(program_cell2, tf.tile(x22[:, -1:, :], [1, self.L2, 1]), x12, x22, h22,
                                                     I * L, L, self.L2)
                x_p = tf.concat([x_p1, x_p2], axis=-1)



        x1 = tf.layers.dense(x_p, len(impl.FUNCTIONS) + 8, activation=None)
        x2 = tf.reshape(x1, [-1, M, self.L2, len(impl.FUNCTIONS) + 8])
        pooled = tf.layers.max_pooling2d(x2, [M, 1], [1,1])

        pred = tf.reshape(pooled, [-1, self.L2, len(impl.FUNCTIONS) + 8])

        self.pred = tf.nn.softmax(pred, axis=-1)

        with tf.name_scope('train_loss'):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph, logits=pred))
            # one_hot_lable = tf.one_hot(self.label_ph, len(impl.FUNCTIONS) + 8)
            # self.loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(one_hot_lable * -tf.log(self.pred) + (1-one_hot_lable) * -tf.log(1-self.pred), axis=-1)),axis=-1)

            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.merged = tf.summary.merge_all()

    def fit(self, rows_type, rows_val, y, epochs, validation_split):
        for i in range(epochs):
            if self.batch_size==-1:
                print('start epochs', i)
                _, summary, loss = self.sess.run([self.train_op, self.merged, self.loss], feed_dict={self.type_ph: rows_type,
                                                                                    self.val_ph: rows_val,
                                                                                    self.label_ph: y})
                self.writer.add_summary(summary, i)
                print('loss:', loss)
            else:
                print('start epochs', i)
                # zipped = zip(rows_type, rows_val, y)
                # print(*zipped)
                # random.shuffle(zipped)
                # rows_type, rows_val, y = zipped
                for j in range(0, len(y), self.batch_size):
                    _, summary, loss = self.sess.run([self.train_op, self.merged, self.loss],
                                                     feed_dict={self.type_ph: rows_type[j:j+self.batch_size],
                                                                self.val_ph: rows_val[j:j+self.batch_size],
                                                                self.label_ph: y[j:j+self.batch_size]})
                self.writer.add_summary(summary, i)
                print('loss:', loss)

    def save(self, outfile="../models/rubustfill/model.ckpt"):
        self.saver.save(self.sess, outfile)

    def load(self, outfile="models/deepcoder.ckpt"):
        ckpt = tf.train.get_checkpoint_state("../models/rubustfill/")
        if ckpt and ckpt.model_checkpoint_path:
            print('load successful')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def predict(self, rows_type, rows_val):
        pred = self.sess.run(self.pred, feed_dict={self.type_ph: rows_type, self.val_ph: rows_val})
        return pred

    def attention_rnn(self, cell, x_input, xs1, h0, in_len, out_len):
        x_list = []
        # xs1 = [b*M, L2, dim]
        # h0 = [2, b*M, dim]
        h = h0
        for i in range(out_len):
            q = h[0]
            # print(q.shape)
            e = tf.tile(tf.reshape(q, [-1, 1, self.dim]), [1, in_len, 1]) * xs1  # [b*M, in_len, dim]
            # print(e.shape)
            a = tf.nn.softmax(tf.reduce_sum(e, axis=-1), axis=-1)  # [b*M, in_len]
            # print(a.shape)
            new_x = tf.reshape(tf.matmul(tf.reshape(a, [-1, 1, in_len]), xs1),[-1, self.dim])  # [b*M, dim]

            new_input = tf.concat([new_x, x_input[:, i, :]], axis=-1)

            x, h = cell(new_input, h)
            # print(x.shape)
            x_list.append(x)
            # print(1111)

        # x_list = [out_len, ?, dim]
        x_out = tf.stack(x_list,axis=1)
        # print(x_out.shape)

        return x_out, h

    def double_attention_rnn(self, cell, x_input, xs1, xs2, h0, in_len1, in_len2, out_len):
        x_list = []
        # xs1 = [b*M, L2, dim]
        # h0 = [2, b*M, dim]
        h = h0
        for i in range(out_len):
            q = h[0]
            # print(q.shape)
            e1 = tf.tile(tf.reshape(q, [-1, 1, self.dim]), [1, in_len1, 1]) * xs1  # [b*M, in_len, dim]
            e2 = tf.tile(tf.reshape(q, [-1, 1, self.dim]), [1, in_len2, 1]) * xs2  # [b*M, in_len, dim]
            # print(e.shape)
            a1 = tf.nn.softmax(tf.reduce_sum(e1, axis=-1), axis=-1)  # [b*M, in_len]
            a2 = tf.nn.softmax(tf.reduce_sum(e2, axis=-1), axis=-1)  # [b*M, in_len]
            # print(a.shape)
            new_x1 = tf.reshape(tf.matmul(tf.reshape(a1, [-1, 1, in_len1]), xs1),[-1, self.dim])  # [b*M, dim]
            new_x2 = tf.reshape(tf.matmul(tf.reshape(a2, [-1, 1, in_len2]), xs2), [-1, self.dim])  # [b*M, dim]

            new_input = tf.concat([new_x1, new_x2, x_input[:, i, :]], axis=-1)

            x, h = cell(new_input, h)
            # print(x.shape)
            x_list.append(x)
            # print(1111)

        # x_list = [out_len, ?, dim]
        x_out = tf.stack(x_list,axis=1)
        # print(x_out.shape)

        return x_out, h
