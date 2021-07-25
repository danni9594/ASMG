from __future__ import print_function
from __future__ import division
import tensorflow as tf


def gelu(input_tensor):
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def transfer(name, prev, upd, n1=10, n2=5, l1=20):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        dim = upd.get_shape().as_list()[-1]  # H
        norm = tf.sqrt(tf.reduce_sum(prev * prev, axis=-1))  # [num]
        dot = tf.div(prev * upd, tf.expand_dims(norm, -1) + tf.constant(1e-15))  # [num, H]
        stack = tf.stack([prev, upd, dot], axis=1)  # [num, 3, H]

        input1 = tf.expand_dims(stack, -1)  # [num, 3, H, 1]
        filter1 = tf.get_variable(name="cnn_filter1", shape=[3, 1, 1, n1])  # [3, 1, 1, n1]
        output1 = tf.nn.conv2d(input1, filter1, strides=[1, 1, 1, 1], padding='VALID')  # [num, 1, H, n1]
        output1 = gelu(output1)  # [num, 1, H, n1]

        input2 = tf.transpose(output1, perm=[0, 3, 2, 1])  # [num, n1, H, 1]
        filter2 = tf.get_variable(name="cnn_filter2", shape=[n1, 1, 1, n2])  # [n1, 1, 1, n2]
        output2 = tf.nn.conv2d(input2, filter2, strides=[1, 1, 1, 1], padding='VALID')  # [num, 1, H, n2]
        output2 = gelu(output2)  # [num, 1, H, n2]

        cnn_output = tf.transpose(output2, perm=[0, 3, 2, 1])  # [num, n2, H, 1]
        cnn_output = tf.reshape(cnn_output, shape=[-1, n2 * dim])  # [num, n2 x H]

        with tf.variable_scope('fcn1'):
            fcn1_kernel = tf.get_variable(name='kernel', shape=[n2 * dim, l1])  # [n2 x H, l1]
            fcn1_bias = tf.get_variable(name='bias', shape=[l1])  # [l1]
        with tf.variable_scope('fcn2'):
            fcn2_kernel = tf.get_variable(name='kernel', shape=[l1, dim])  # [l1, H]
            fcn2_bias = tf.get_variable(name='bias', shape=[dim])  # [H]

        fcn1 = gelu(tf.matmul(cnn_output, fcn1_kernel) + fcn1_bias)  # [num, l1]
        fcn2 = tf.matmul(fcn1, fcn2_kernel) + fcn2_bias  # [num, H]

    return fcn2


class SMLmf(object):

    def __init__(self, hyperparams, prev_emb_dict, prev_bias_dict, train_config=None):

        self.train_config = train_config

        # create placeholder
        self.u = tf.placeholder(tf.int32, [None])  # [B]
        self.i = tf.placeholder(tf.int32, [None])  # [B]
        self.y = tf.placeholder(tf.float32, [None])  # [B]
        self.base_lr = tf.placeholder(tf.float32, [], name='base_lr')  # scalar
        self.transfer_lr = tf.placeholder(tf.float32, [], name='transfer_lr')  # scalar

        # -- create emb_upd begin -------
        user_emb_w_upd = tf.get_variable("user_emb_w", [hyperparams['num_users'], hyperparams['user_embed_dim']])
        item_emb_w_upd = tf.get_variable("item_emb_w", [hyperparams['num_items'], hyperparams['item_embed_dim']])
        # -- create emb_upd end -------

        # -- create bias_upd begin ---
        user_b_upd = tf.get_variable("user_b", [hyperparams['num_users']], initializer=tf.constant_initializer(0.0))
        item_b_upd = tf.get_variable("item_b", [hyperparams['num_items']], initializer=tf.constant_initializer(0.0))
        # -- create bias_upd end ---

        # -- create emb_prev begin ----
        user_emb_w_prev = tf.convert_to_tensor(prev_emb_dict['user_emb_w'], tf.float32)
        item_emb_w_prev = tf.convert_to_tensor(prev_emb_dict['item_emb_w'], tf.float32)
        # -- create emb_prev end ----

        # -- create bias_prev begin ----
        user_b_prev = tf.convert_to_tensor(prev_bias_dict['user_b'], tf.float32)
        item_b_prev = tf.convert_to_tensor(prev_bias_dict['item_b'], tf.float32)
        # -- create bias_prev end ----

        # -- concat emb and bias begin ----
        user_w_upd = tf.concat([user_emb_w_upd, tf.expand_dims(user_b_upd, -1)], -1)  # [num_users, user_embed_dim + 1]
        item_w_upd = tf.concat([item_emb_w_upd, tf.expand_dims(item_b_upd, -1)], -1)  # [num_items, item_embed_dim + 1]
        user_w_prev = tf.concat([user_emb_w_prev, tf.expand_dims(user_b_prev, -1)], -1)  # [num_users, user_embed_dim + 1]
        item_w_prev = tf.concat([item_emb_w_prev, tf.expand_dims(item_b_prev, -1)], -1)  # [num_items, item_embed_dim + 1]
        # -- concat emb and bias end ----

        # -- transfer emb begin ----
        with tf.variable_scope('transfer'):
            user_w = transfer(name='user_w',
                              prev=user_w_prev,
                              upd=user_w_upd,
                              n1=train_config['n1'],
                              n2=train_config['n2'],
                              l1=train_config['l1'])
            item_w = transfer(name='item_w',
                              prev=item_w_prev,
                              upd=item_w_upd,
                              n1=train_config['n1'],
                              n2=train_config['n2'],
                              l1=train_config['l1'])
            user_emb_w = user_w[:, :-1]  # [num_users, user_embed_dim]
            user_b = user_w[:, -1]  # [num_users]
            item_emb_w = item_w[:, :-1]  # [num_items, item_embed_dim]
            item_b = item_w[:, -1]  # [num_items]
        # -- transfer emb end ----

        # -- update op begin -------
        self.user_emb_w_upd_op = user_emb_w_upd.assign(user_emb_w)
        self.user_b_upd_op = user_b_upd.assign(user_b)
        self.item_emb_w_upd_op = item_emb_w_upd.assign(item_emb_w)
        self.item_b_upd_op = item_b_upd.assign(item_b)
        # -- update op end -------

        # -- emb begin -------
        u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)  # [B, H]
        i_emb = tf.nn.embedding_lookup(item_emb_w, self.i)  # [B, H]
        # -- emb end -------

        # -- bias begin ---
        u_b = tf.gather(user_b, self.u)  # [B]
        i_b = tf.gather(item_b, self.i)  # [B]
        # -- bias end ---

        interaction = tf.reduce_sum(u_emb * i_emb, axis=-1)  # [B]

        logits = interaction + u_b + i_b  # [B]
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

        # transfer_optimizer
        if train_config['transfer_optimizer'] == 'adam':
            transfer_optimizer = tf.train.AdamOptimizer(learning_rate=self.transfer_lr)
        elif train_config['transfer_optimizer'] == 'rmsprop':
            transfer_optimizer = tf.train.RMSPropOptimizer(learning_rate=self.transfer_lr)
        else:
            transfer_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.transfer_lr)

        trainable_params = tf.trainable_variables()
        base_params = [v for v in trainable_params if 'transfer' not in v.name]
        transfer_params = [v for v in trainable_params if 'transfer' in v.name]

        # update base model and transfer module
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            base_grads = tf.gradients(self.loss, base_params)  # return a list of gradients (A list of `sum(dy/dx)` for each x in `xs`)
            base_grads_tuples = zip(base_grads, base_params)
            self.train_base_op = base_optimizer.apply_gradients(base_grads_tuples)

            transfer_grads = tf.gradients(self.loss, transfer_params)
            transfer_grads_tuples = zip(transfer_grads, transfer_params)
            with tf.variable_scope('transfer_opt'):
                self.train_transfer_op = transfer_optimizer.apply_gradients(transfer_grads_tuples)

    def train_base(self, sess, batch):
        loss, _ = sess.run([self.loss, self.train_base_op], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.y: batch[2],
            self.base_lr: self.train_config['base_lr'],
        })
        return loss

    def train_transfer(self, sess, batch):
        loss, _, = sess.run([self.loss, self.train_transfer_op], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.y: batch[2],
            self.transfer_lr: self.train_config['transfer_lr'],
        })
        return loss

    def update(self, sess):
        sess.run([self.user_emb_w_upd_op, self.user_b_upd_op,
                  self.item_emb_w_upd_op, self.item_b_upd_op])

    def inference(self, sess, batch):
        scores, losses = sess.run([self.scores, self.losses], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.y: batch[2],
        })
        return scores, losses
