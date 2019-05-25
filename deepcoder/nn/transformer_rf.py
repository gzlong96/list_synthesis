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

from deepcoder.nn.transformer_modules import ff, positional_encoding, multihead_attention

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


class Transfill:
    def __init__(self, I, E, L2, K=256, lr=1e-3, batch_size=-1, nb_tf=(3,3,3), nb_head=4):
        self.I = I
        self.E = E
        self.L2 = L2  # max length of tokens
        self.dim = K

        self.lr = lr
        self.batch_size = batch_size
        self.nb_tf = nb_tf
        self.nb_head = nb_head

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
        program_embeddings = tf.get_variable('program_embeddings', [len(impl.ACT_SPACE), self.dim])

        embedded_vals = tf.nn.embedding_lookup(number_embeddings, self.val_ph)
        embedded_program = tf.nn.embedding_lookup(program_embeddings, self.label_ph)


        concated = tf.concat([self.type_ph, embedded_vals], axis=-1)  # [b, M, I+1, L, E+2]

        reshaped_i_vals = tf.reshape(concated[:,:,:I,:,:], [-1, I * L, E + 2]) # [b*M, I * L, E + 2]
        reshaped_o_vals = tf.reshape(concated[:,:,I:,:,:], [-1, L, E + 2])  # [b*M, L, E + 2]

        reshaped_i_vals += positional_encoding(reshaped_i_vals, I * L)
        reshaped_o_vals += positional_encoding(reshaped_o_vals, L)


        with tf.name_scope('i_encoder'):
            enc1 = reshaped_i_vals
            for i in range(self.nb_tf[0]):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc1 = multihead_attention(queries=enc1,
                                              keys=enc1,
                                              values=enc1,
                                              num_heads=self.nb_head,
                                              dropout_rate=0,
                                              training=True,
                                              causality=False)
                    # feed forward
                    enc1 = ff(enc1, num_units=[self.dim, self.dim])

        with tf.name_scope('o_encoder'):
            with tf.variable_scope("encoder2", reuse=tf.AUTO_REUSE):
                # embedding
                enc2 = reshaped_o_vals
                # Blocks
                for i in range(self.nb_tf[1]):
                    with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                        # Masked self-attention (Note that causality is True at this time)
                        enc2 = multihead_attention(queries=enc2,
                                                  keys=enc2,
                                                  values=enc2,
                                                  num_heads=self.nb_head,
                                                  dropout_rate=0,
                                                  training=True,
                                                  causality=False,
                                                  scope="self_attention")

                        # Vanilla attention
                        enc2 = multihead_attention(queries=enc2,
                                                  keys=enc1,
                                                  values=enc1,
                                                  num_heads=self.nb_head,
                                                  dropout_rate=0,
                                                  training=True,
                                                  causality=False,
                                                  scope="vanilla_attention")
                        ### Feed Forward
                        enc2 = ff(enc2, num_units=[self.dim, self.dim])

        with tf.name_scope('program_decoder'):
            with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
                # embedding
                dec = tf.reshape(tf.tile(tf.expand_dims(embedded_program,axis=1), [1, M, 1, 1]), [-1, self.L2, self.dim])
                # dec = tf.ones_like(tf.reshape(tf.tile(tf.expand_dims(embedded_program, axis=1), [1, M, 1, 1]),
                #                  [-1, self.L2, self.dim]))
                # Blocks
                for i in range(self.nb_tf[1]):
                    with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                        # Masked self-attention (Note that causality is True at this time)
                        dec = multihead_attention(queries=dec,
                                                  keys=dec,
                                                  values=dec,
                                                  num_heads=self.nb_head,
                                                  dropout_rate=0,
                                                  training=True,
                                                  causality=True,
                                                  scope="self_attention")

                        # Vanilla attention
                        dec = multihead_attention(queries=dec,
                                                  keys=enc2,
                                                  values=enc2,
                                                  num_heads=self.nb_head,
                                                  dropout_rate=0,
                                                  training=True,
                                                  causality=False,
                                                  scope="vanilla_attention")
                        ### Feed Forward
                        dec = ff(dec, num_units=[self.dim, self.dim])

        weights = tf.transpose(program_embeddings)  # (d_model, vocab_size)
        x1 = tf.reshape(dec, [-1, M, self.L2, self.dim])
        pooled = tf.reshape(tf.layers.max_pooling2d(x1, [M, 1], [1, 1]), [-1, self.L2, self.dim])

        pred = tf.einsum('ntd,dk->ntk', pooled, weights)  # (N, T2, vocab_size)


        # x1 = tf.layers.dense(dec, len(impl.ACT_SPACE), activation=None)
        # x2 = tf.reshape(x1, [-1, M, self.L2, len(impl.ACT_SPACE)])
        # pooled = tf.layers.max_pooling2d(x2, [M, 1], [1,1])
        #
        # pred = tf.reshape(pooled, [-1, self.L2, len(impl.ACT_SPACE)])

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

    def save(self, outfile="../models/transfill/model.ckpt"):
        self.saver.save(self.sess, outfile)

    def load(self, outfile="models/deepcoder.ckpt"):
        ckpt = tf.train.get_checkpoint_state("../models/transfill/")
        if ckpt and ckpt.model_checkpoint_path:
            print('load successful')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def predict(self, rows_type, rows_val, fake_y):
        pred = self.sess.run(self.pred, feed_dict={self.type_ph: rows_type, self.val_ph: rows_val, self.label_ph:fake_y})
        return pred

