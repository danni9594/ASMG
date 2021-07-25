from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


def average_pooling(emb, seq_len):
    mask = tf.sequence_mask(seq_len, tf.shape(emb)[-2], dtype=tf.float32)  # [B, T] / [B, T, max_cate_len]
    mask = tf.expand_dims(mask, -1)  # [B, T, 1] / [B, T, max_cate_len, 1]
    emb *= mask  # [B, T, H] / [B, T, max_cate_len, H]
    sum_pool = tf.reduce_sum(emb, -2)  # [B, H] / [B, T, H]
    avg_pool = tf.div(sum_pool, tf.expand_dims(tf.cast(seq_len, tf.float32), -1) + 1e-8)  # [B, H] / [B, T, H]
    return avg_pool


def linear_combine(name, param_ls, param_shape, seq_length):
    """
    perform linear weighted sum combination on sequence of parameters and output the final w and output
    :param name: param name
    :param param_ls: param list of seq_length
    :param param_shape: param shape [d1, d2, ...]
    :param seq_length: input sequence length
    :return: final w [d1, d2, ..., seq_length], final output [d1, d2, ..., seq_length]
    """
    with tf.variable_scope('linear_combine'):
        if 'fcn' in name:
            w = tf.nn.softmax(tf.get_variable(name=name + '_w',
                                              shape=param_shape + [seq_length],
                                              initializer=tf.constant_initializer(0)))
        else:  # 'emb' in name
            w = tf.nn.softmax(tf.get_variable(name=name + '_w',
                                              shape=param_shape[1:] + [seq_length],
                                              initializer=tf.constant_initializer(0)))
        output = tf.reduce_sum(param_ls * w, -1)  # weighted sum of the last dimension
    return w, output


class ASMGlinear(object):

    def __init__(self, cates, cate_lens, hyperparams, emb_ls_dict, mlp_ls_dict, train_config=None):

        self.train_config = train_config

        # create placeholder
        self.u = tf.placeholder(tf.int32, [None])  # [B]
        self.i = tf.placeholder(tf.int32, [None])  # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.hist_len = tf.placeholder(tf.int32, [None])  # [B]
        self.y = tf.placeholder(tf.float32, [None])  # [B]
        self.meta_lr = tf.placeholder(tf.float32, [], name='meta_lr')  # scalar

        cates = tf.convert_to_tensor(cates, dtype=tf.int32)  # [num_cates, max_cate_len]
        cate_lens = tf.convert_to_tensor(cate_lens, dtype=tf.int32)  # [num_cates]

        # -- create emb_w_ls begin ----
        user_emb_w_ls = tf.convert_to_tensor(emb_ls_dict['user_emb_w'], tf.float32)  # [num_users, user_embed_dim, seq_length]
        item_emb_w_ls = tf.convert_to_tensor(emb_ls_dict['item_emb_w'], tf.float32)  # [num_items, item_embed_dim, seq_length]
        cate_emb_w_ls = tf.convert_to_tensor(emb_ls_dict['cate_emb_w'], tf.float32)  # [num_cates, cate_embed_dim, seq_length]
        # -- create emb_w_ls end ----

        # -- create mlp_ls begin ----
        fcn1_kernel_ls = tf.convert_to_tensor(mlp_ls_dict['fcn1/kernel'], tf.float32)  # [concat_dim, l1, seq_length]
        fcn1_bias_ls = tf.convert_to_tensor(mlp_ls_dict['fcn1/bias'], tf.float32)  # [l1, seq_length]
        fcn2_kernel_ls = tf.convert_to_tensor(mlp_ls_dict['fcn2/kernel'], tf.float32)  # [l1, l2, seq_length]
        fcn2_bias_ls = tf.convert_to_tensor(mlp_ls_dict['fcn2/bias'], tf.float32)  # [l2, seq_length]
        fcn3_kernel_ls = tf.convert_to_tensor(mlp_ls_dict['fcn3/kernel'], tf.float32)  # [l2, 1, seq_length]
        fcn3_bias_ls = tf.convert_to_tensor(mlp_ls_dict['fcn3/bias'], tf.float32)  # [1, seq_length]
        # -- create mlp_ls end ----

        # -- generate emb_w begin ----
        with tf.variable_scope('meta_emb'):
            self.user_emb_w_w, user_emb_w = linear_combine(name='user_emb_w',
                                                           param_ls=user_emb_w_ls,
                                                           param_shape=[hyperparams['num_users'], hyperparams['user_embed_dim']],
                                                           seq_length=train_config['seq_length'])
            self.item_emb_w_w, item_emb_w = linear_combine(name='item_emb_w',
                                                           param_ls=item_emb_w_ls,
                                                           param_shape=[hyperparams['num_items'], hyperparams['item_embed_dim']],
                                                           seq_length=train_config['seq_length'])
            self.cate_emb_w_w, cate_emb_w = linear_combine(name='cate_emb_w',
                                                           param_ls=cate_emb_w_ls,
                                                           param_shape=[hyperparams['num_cates'], hyperparams['cate_embed_dim']],
                                                           seq_length=train_config['seq_length'])
        # -- generate emb_w end ----

        # -- generate mlp begin ----
        concat_dim = hyperparams['user_embed_dim'] + (hyperparams['item_embed_dim'] + hyperparams['cate_embed_dim']) * 2
        with tf.variable_scope('meta_emb'):
            self.fcn1_kernel_w, fcn1_kernel = linear_combine(name='fcn1_kernel',
                                                             param_ls=fcn1_kernel_ls,
                                                             param_shape=[concat_dim, hyperparams['layers'][1]],
                                                             seq_length=train_config['seq_length'])
            self.fcn1_bias_w, fcn1_bias = linear_combine(name='fcn1_bias',
                                                         param_ls=fcn1_bias_ls,
                                                         param_shape=[hyperparams['layers'][1]],
                                                         seq_length=train_config['seq_length'])
            self.fcn2_kernel_w, fcn2_kernel = linear_combine(name='fcn2_kernel',
                                                             param_ls=fcn2_kernel_ls,
                                                             param_shape=[hyperparams['layers'][1], hyperparams['layers'][2]],
                                                             seq_length=train_config['seq_length'])
            self.fcn2_bias_w, fcn2_bias = linear_combine(name='fcn2_bias',
                                                         param_ls=fcn2_bias_ls,
                                                         param_shape=[hyperparams['layers'][2]],
                                                         seq_length=train_config['seq_length'])
            self.fcn3_kernel_w, fcn3_kernel = linear_combine(name='fcn3_kernel',
                                                             param_ls=fcn3_kernel_ls,
                                                             param_shape=[hyperparams['layers'][2], 1],
                                                             seq_length=train_config['seq_length'])
            self.fcn3_bias_w, fcn3_bias = linear_combine(name='fcn3_bias',
                                                         param_ls=fcn3_bias_ls,
                                                         param_shape=[1],
                                                         seq_length=train_config['seq_length'])
        # -- generate mlp end ----

        # -- emb begin -------
        u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)  # [B, H]

        ic = tf.gather(cates, self.i)  # [B, max_cate_len]
        ic_len = tf.gather(cate_lens, self.i)  # [B]
        i_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.i),
            average_pooling(tf.nn.embedding_lookup(cate_emb_w, ic), ic_len)
        ], axis=1)  # [B, H x 2]

        hist_c = tf.gather(cates, self.hist_i)  # [B, T, max_cate_len]
        hist_c_len = tf.gather(cate_lens, self.hist_i)  # [B, T]
        h_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.hist_i),
            average_pooling(tf.nn.embedding_lookup(cate_emb_w, hist_c), hist_c_len)
        ], axis=2)  # [B, T, H x 2]
        u_hist = average_pooling(h_emb, self.hist_len)  # [B, H x 2]
        # -- emb end -------

        # -- mlp begin -------
        fcn = tf.concat([u_emb, u_hist, i_emb], axis=-1)  # [B, H x 5]
        fcn_layer_1 = tf.nn.relu(tf.matmul(fcn, fcn1_kernel) + fcn1_bias)  # [B, l1]
        fcn_layer_2 = tf.nn.relu(tf.matmul(fcn_layer_1, fcn2_kernel) + fcn2_bias)  # [B, l2]
        fcn_layer_3 = tf.matmul(fcn_layer_2, fcn3_kernel) + fcn3_bias  # [B, 1]
        # -- mlp end -------

        logits = tf.reshape(fcn_layer_3, [-1])  # [B]
        self.scores = tf.sigmoid(logits)  # [B]

        # return same dimension as input tensors, let x = logits, z = labels, z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y)
        self.loss = tf.reduce_mean(self.losses)  # for training loss

        # meta_optimizer
        if train_config['meta_optimizer'] == 'adam':
            meta_optimizer = tf.train.AdamOptimizer(learning_rate=self.meta_lr)
        elif train_config['meta_optimizer'] == 'rmsprop':
            meta_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.meta_lr)
        else:
            meta_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.meta_lr)

        trainable_params = tf.trainable_variables()

        # update meta generator
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            meta_grads = tf.gradients(self.loss, trainable_params)
            meta_grads_tuples = zip(meta_grads, trainable_params)
            with tf.variable_scope('meta_opt'):
                self.train_meta_op = meta_optimizer.apply_gradients(meta_grads_tuples)

    def train_meta(self, sess, batch, batch_id):
        loss, _, user_emb_w_w, item_emb_w_w, cate_emb_w_w, \
        fcn1_kernel_w, fcn1_bias_w, fcn2_kernel_w, fcn2_bias_w, \
        fcn3_kernel_w, fcn3_bias_w = sess.run([
            self.loss, self.train_meta_op, self.user_emb_w_w, self.item_emb_w_w, self.cate_emb_w_w,
            self.fcn1_kernel_w, self.fcn1_bias_w, self.fcn2_kernel_w, self.fcn2_bias_w,
            self.fcn3_kernel_w, self.fcn3_bias_w], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[2],
            self.hist_len: batch[3],
            self.y: batch[4],
            self.meta_lr: self.train_config['meta_lr'],
        })
        if (batch_id - 1) % 100 == 0:
            for i in range(self.train_config['seq_length']):
                print('period {} '.format(i), '-' * 60)
                print('user_emb_w_w: mean', np.mean(user_emb_w_w[..., i]), 'std', np.std(user_emb_w_w[..., i]), 'min', np.min(user_emb_w_w[..., i]), 'max', np.max(user_emb_w_w[..., i]))
                print('item_emb_w_w: mean', np.mean(item_emb_w_w[..., i]), 'std', np.std(item_emb_w_w[..., i]), 'min', np.min(item_emb_w_w[..., i]), 'max', np.max(item_emb_w_w[..., i]))
                print('cate_emb_w_w: mean', np.mean(cate_emb_w_w[..., i]), 'std', np.std(cate_emb_w_w[..., i]), 'min', np.min(cate_emb_w_w[..., i]), 'max', np.max(cate_emb_w_w[..., i]))
                print('')
                print('fcn1_kernel_w: mean', np.mean(fcn1_kernel_w[..., i]), 'std', np.std(fcn1_kernel_w[..., i]), 'min', np.min(fcn1_kernel_w[..., i]), 'max', np.max(fcn1_kernel_w[..., i]))
                print('fcn1_bias_w: mean', np.mean(fcn1_bias_w[..., i]), 'std', np.std(fcn1_bias_w[..., i]), 'min', np.min(fcn1_bias_w[..., i]), 'max', np.max(fcn1_bias_w[..., i]))
                print('fcn2_kernel_w: mean', np.mean(fcn2_kernel_w[..., i]), 'std', np.std(fcn2_kernel_w[..., i]), 'min', np.min(fcn2_kernel_w[..., i]), 'max', np.max(fcn2_kernel_w[..., i]))
                print('fcn2_bias_w: mean', np.mean(fcn2_bias_w[..., i]), 'std', np.std(fcn2_bias_w[..., i]), 'min', np.min(fcn2_bias_w[..., i]), 'max', np.max(fcn2_bias_w[..., i]))
                print('fcn3_kernel_w: mean', np.mean(fcn3_kernel_w[..., i]), 'std', np.std(fcn3_kernel_w[..., i]), 'min', np.min(fcn3_kernel_w[..., i]), 'max', np.max(fcn3_kernel_w[..., i]))
                print('fcn3_bias_w: mean', np.mean(fcn3_bias_w[..., i]), 'std', np.std(fcn3_bias_w[..., i]), 'min', np.min(fcn3_bias_w[..., i]), 'max', np.max(fcn3_bias_w[..., i]))
                print('')
        return loss

    def inference(self, sess, batch):
        scores, losses = sess.run([self.scores, self.losses], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[2],
            self.hist_len: batch[3],
            self.y: batch[4],
        })
        return scores, losses
