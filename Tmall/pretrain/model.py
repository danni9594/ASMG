from __future__ import print_function
from __future__ import division
import tensorflow as tf


def average_pooling(emb, seq_len):
    mask = tf.sequence_mask(seq_len, tf.shape(emb)[-2], dtype=tf.float32)  # [B, T] / [B, T, max_cate_len]
    mask = tf.expand_dims(mask, -1)  # [B, T, 1] / [B, T, max_cate_len, 1]
    emb *= mask  # [B, T, H] / [B, T, max_cate_len, H]
    sum_pool = tf.reduce_sum(emb, -2)  # [B, H] / [B, T, H]
    avg_pool = tf.div(sum_pool, tf.expand_dims(tf.cast(seq_len, tf.float32), -1) + 1e-8)  # [B, H] / [B, T, H]
    return avg_pool


class EmbMLP(object):
    """
        Embedding&MLP base model
    """
    def __init__(self, cates, cate_lens, hyperparams, train_config=None):

        self.train_config = train_config

        # create placeholder
        self.u = tf.placeholder(tf.int32, [None])  # [B]
        self.i = tf.placeholder(tf.int32, [None])  # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.hist_len = tf.placeholder(tf.int32, [None])  # [B]
        self.y = tf.placeholder(tf.float32, [None])  # [B]
        self.base_lr = tf.placeholder(tf.float32, [], name='base_lr')  # scalar

        cates = tf.convert_to_tensor(cates, dtype=tf.int32)  # [num_cates, max_cate_len]
        cate_lens = tf.convert_to_tensor(cate_lens, dtype=tf.int32)  # [num_cates]

        # -- create emb begin -------
        user_emb_w = tf.get_variable("user_emb_w", [hyperparams['num_users'], hyperparams['user_embed_dim']])
        item_emb_w = tf.get_variable("item_emb_w", [hyperparams['num_items'], hyperparams['item_embed_dim']])
        cate_emb_w = tf.get_variable("cate_emb_w", [hyperparams['num_cates'], hyperparams['cate_embed_dim']])
        # -- create emb end -------

        # -- create mlp begin ---
        concat_dim = hyperparams['user_embed_dim'] + (hyperparams['item_embed_dim'] + hyperparams['cate_embed_dim']) * 2
        with tf.variable_scope('fcn1'):
            fcn1_kernel = tf.get_variable(name='kernel', shape=[concat_dim, hyperparams['layers'][1]])
            fcn1_bias = tf.get_variable(name='bias', shape=[hyperparams['layers'][1]])
        with tf.variable_scope('fcn2'):
            fcn2_kernel = tf.get_variable(name='kernel', shape=[hyperparams['layers'][1], hyperparams['layers'][2]])
            fcn2_bias = tf.get_variable(name='bias', shape=[hyperparams['layers'][2]])
        with tf.variable_scope('fcn3'):
            fcn3_kernel = tf.get_variable(name='kernel', shape=[hyperparams['layers'][2], 1])
            fcn3_bias = tf.get_variable(name='bias', shape=[1])
        # -- create mlp end ---

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
        self.loss = tf.reduce_mean(self.losses)

        # base_optimizer
        if train_config['base_optimizer'] == 'adam':
            base_optimizer = tf.train.AdamOptimizer(learning_rate=self.base_lr)
        elif train_config['base_optimizer'] == 'rmsprop':
            base_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.base_lr)
        else:
            base_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.base_lr)

        trainable_params = tf.trainable_variables()

        # update base model
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            base_grads = tf.gradients(self.loss, trainable_params)  # return a list of gradients (A list of `sum(dy/dx)` for each x in `xs`)
            base_grads_tuples = zip(base_grads, trainable_params)
            self.train_base_op = base_optimizer.apply_gradients(base_grads_tuples)

    def train_base(self, sess, batch):
        loss, _ = sess.run([self.loss, self.train_base_op], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[2],
            self.hist_len: batch[3],
            self.y: batch[4],
            self.base_lr: self.train_config['base_lr'],
        })
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
