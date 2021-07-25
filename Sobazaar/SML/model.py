from __future__ import print_function
from __future__ import division
import tensorflow as tf


def average_pooling(emb, seq_len):
    mask = tf.sequence_mask(seq_len, tf.shape(emb)[-2], dtype=tf.float32)  # [B, T]
    mask = tf.expand_dims(mask, -1)  # [B, T, 1]
    emb *= mask  # [B, T, H]
    sum_pool = tf.reduce_sum(emb, -2)  # [B, H]
    avg_pool = tf.div(sum_pool, tf.expand_dims(tf.cast(seq_len, tf.float32), -1) + 1e-8)  # [B, H]
    return avg_pool


def gelu(input_tensor):
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def transfer_emb(name, emb_prev, emb_upd, n1=10, n2=5, l1=20):

    with tf.variable_scope(name):
        embed_dim = emb_upd.get_shape().as_list()[-1]  # H
        embeds_norm = tf.sqrt(tf.reduce_sum(emb_prev * emb_prev, axis=-1))  # [num]
        embeds_dot = tf.div(emb_prev * emb_upd, tf.expand_dims(embeds_norm, -1) + tf.constant(1e-15))  # [num, H]
        stack_embeds = tf.stack([emb_prev, emb_upd, embeds_dot], axis=1)  # [num, 3, H]

        input1 = tf.expand_dims(stack_embeds, -1)  # [num, 3, H, 1]
        filter1 = tf.get_variable(name="cnn_filter1", shape=[3, 1, 1, n1])  # [3, 1, 1, n1]
        output1 = tf.nn.conv2d(input1, filter1, strides=[1, 1, 1, 1], padding='VALID')  # [num, 1, H, n1]
        output1 = gelu(output1)  # [num, 1, H, n1]

        input2 = tf.transpose(output1, perm=[0, 3, 2, 1])  # [num, n1, H, 1]
        filter2 = tf.get_variable(name="cnn_filter2", shape=[n1, 1, 1, n2])  # [n1, 1, 1, n2]
        output2 = tf.nn.conv2d(input2, filter2, strides=[1, 1, 1, 1], padding='VALID')  # [num, 1, H, n2]
        output2 = gelu(output2)  # [num, 1, H, n2]

        cnn_output = tf.transpose(output2, perm=[0, 3, 2, 1])  # [num, n2, H, 1]
        cnn_output = tf.reshape(cnn_output, shape=[-1, n2 * embed_dim])  # [num, n2 x H]

        with tf.variable_scope('fcn1'):
            fcn1_kernel = tf.get_variable(name='kernel', shape=[n2 * embed_dim, l1])  # [n2 x H, l1]
            fcn1_bias = tf.get_variable(name='bias', shape=[l1])  # [l1]
        with tf.variable_scope('fcn2'):
            fcn2_kernel = tf.get_variable(name='kernel', shape=[l1, embed_dim])  # [l1, H]
            fcn2_bias = tf.get_variable(name='bias', shape=[embed_dim])  # [H]

        fcn1 = gelu(tf.matmul(cnn_output, fcn1_kernel) + fcn1_bias)  # [num, l1]
        fcn2 = tf.matmul(fcn1, fcn2_kernel) + fcn2_bias  # [num, H]

    return fcn2


def transfer_mlp(name, param_prev, param_upd, param_shape, n1=5, n2=3, l1=40):

    with tf.variable_scope(name):
        param_prev = tf.reshape(param_prev, [-1])  # [dim]
        param_upd = tf.reshape(param_upd, [-1])  # [dim]
        param_dim = param_upd.get_shape().as_list()[-1]  # max_dim: 24 x 12 = 288
        param_norm = tf.sqrt(tf.reduce_sum(param_prev * param_prev))  # scalar
        param_dot = tf.div(param_prev * param_upd, param_norm + tf.constant(1e-15))  # [dim] / [] = [dim]
        stack_param = tf.stack([param_prev, param_upd, param_dot], axis=0)  # [3, dim]

        input1 = tf.expand_dims(tf.expand_dims(stack_param, -1), 0)  # [1, 3, dim, 1]
        filter1 = tf.get_variable(name="cnn_filter1", shape=[3, 1, 1, n1])  # [3, 1, 1, n1]
        output1 = tf.nn.conv2d(input1, filter1, strides=[1, 1, 1, 1], padding='VALID')  # [1, 1, dim, n1]
        output1 = gelu(output1)  # [1, 1, dim, n1]

        input2 = tf.transpose(output1, perm=[0, 3, 2, 1])  # [1, n1, dim, 1]
        filter2 = tf.get_variable(name="cnn_filter2", shape=[n1, 1, 1, n2])  # [n1, 1, 1, n2]
        output2 = tf.nn.conv2d(input2, filter2, strides=[1, 1, 1, 1], padding='VALID')  # [1, 1, dim, n2]
        output2 = gelu(output2)  # [1, 1, dim, n2]

        cnn_output = tf.transpose(output2, perm=[0, 3, 2, 1])  # [1, n2, dim, 1]
        cnn_output = tf.reshape(cnn_output, shape=[1, -1])  # [1, n2 x dim]

        with tf.variable_scope('fcn1'):
            fcn1_kernel = tf.get_variable(name='kernel', shape=[n2 * param_dim, l1])  # [n2 x dim, l1]
            fcn1_bias = tf.get_variable(name='bias', shape=[l1])  # [l1]
        with tf.variable_scope('fcn2'):
            fcn2_kernel = tf.get_variable(name='kernel', shape=[l1, param_dim])  # [l1, dim]
            fcn2_bias = tf.get_variable(name='bias', shape=[param_dim])  # [dim]

        fcn1 = gelu(tf.matmul(cnn_output, fcn1_kernel) + fcn1_bias)  # [1, l1]
        fcn2 = tf.matmul(fcn1, fcn2_kernel) + fcn2_bias  # [1, dim]
        output = tf.reshape(fcn2, shape=param_shape)  # [dim1, dim2, ...]

    return output


class SML(object):

    def __init__(self, hyperparams, prev_emb_dict, prev_mlp_dict, train_config=None):

        self.train_config = train_config

        # create placeholder
        self.u = tf.placeholder(tf.int32, [None])  # [B]
        self.i = tf.placeholder(tf.int32, [None])  # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.hist_len = tf.placeholder(tf.int32, [None])  # [B]
        self.y = tf.placeholder(tf.float32, [None])  # [B]
        self.base_lr = tf.placeholder(tf.float32, [], name='base_lr')  # scalar
        self.transfer_lr = tf.placeholder(tf.float32, [], name='transfer_lr')  # scalar

        if train_config['transfer_emb']:
            # -- create emb_w_upd begin -------
            user_emb_w_upd = tf.get_variable("user_emb_w", [hyperparams['num_users'], hyperparams['user_embed_dim']])
            item_emb_w_upd = tf.get_variable("item_emb_w", [hyperparams['num_items'], hyperparams['item_embed_dim']])
            # -- create emb_w_upd end -------

            # -- create emb_w_prev begin ----
            user_emb_w_prev = tf.convert_to_tensor(prev_emb_dict['user_emb_w'], tf.float32)
            item_emb_w_prev = tf.convert_to_tensor(prev_emb_dict['item_emb_w'], tf.float32)
            # -- create emb_w_prev end ----

            # -- transfer emb_w begin ----
            with tf.variable_scope('transfer_emb'):
                user_emb_w = transfer_emb(name='user_emb_w',
                                          emb_prev=user_emb_w_prev,
                                          emb_upd=user_emb_w_upd,
                                          n1=train_config['emb_n1'],
                                          n2=train_config['emb_n2'],
                                          l1=train_config['emb_l1'])
                item_emb_w = transfer_emb(name='item_emb_w',
                                          emb_prev=item_emb_w_prev,
                                          emb_upd=item_emb_w_upd,
                                          n1=train_config['emb_n1'],
                                          n2=train_config['emb_n2'],
                                          l1=train_config['emb_l1'])
            # -- transfer emb end ----

            # -- update op begin -------
            self.user_emb_w_upd_op = user_emb_w_upd.assign(user_emb_w)
            self.item_emb_w_upd_op = item_emb_w_upd.assign(item_emb_w)
            # -- update op end -------

        else:
            # -- create emb_w begin -------
            user_emb_w = tf.get_variable("user_emb_w", [hyperparams['num_users'], hyperparams['user_embed_dim']])
            item_emb_w = tf.get_variable("item_emb_w", [hyperparams['num_items'], hyperparams['item_embed_dim']])
            # -- create emb_w end -------

        if train_config['transfer_mlp']:
            # -- create mlp_upd begin ---
            concat_dim = hyperparams['user_embed_dim'] + hyperparams['item_embed_dim'] * 2
            with tf.variable_scope('fcn1'):
                fcn1_kernel_upd = tf.get_variable('kernel', [concat_dim, hyperparams['layers'][1]])
                fcn1_bias_upd = tf.get_variable('bias', [hyperparams['layers'][1]])
            with tf.variable_scope('fcn2'):
                fcn2_kernel_upd = tf.get_variable('kernel', [hyperparams['layers'][1], hyperparams['layers'][2]])
                fcn2_bias_upd = tf.get_variable('bias', [hyperparams['layers'][2]])
            with tf.variable_scope('fcn3'):
                fcn3_kernel_upd = tf.get_variable('kernel', [hyperparams['layers'][2], 1])
                fcn3_bias_upd = tf.get_variable('bias', [1])
            # -- create mlp_upd end ---

            # -- create mlp_prev begin ----
            fcn1_kernel_prev = tf.convert_to_tensor(prev_mlp_dict['fcn1/kernel'], tf.float32)
            fcn1_bias_prev = tf.convert_to_tensor(prev_mlp_dict['fcn1/bias'], tf.float32)
            fcn2_kernel_prev = tf.convert_to_tensor(prev_mlp_dict['fcn2/kernel'], tf.float32)
            fcn2_bias_prev = tf.convert_to_tensor(prev_mlp_dict['fcn2/bias'], tf.float32)
            fcn3_kernel_prev = tf.convert_to_tensor(prev_mlp_dict['fcn3/kernel'], tf.float32)
            fcn3_bias_prev = tf.convert_to_tensor(prev_mlp_dict['fcn3/bias'], tf.float32)
            # -- create mlp_prev end ----

            # -- transfer mlp begin ----
            with tf.variable_scope('transfer_mlp'):
                with tf.variable_scope('fcn1'):
                    fcn1_kernel = transfer_mlp(name='kernel',
                                               param_prev=fcn1_kernel_prev,
                                               param_upd=fcn1_kernel_upd,
                                               param_shape=[concat_dim, hyperparams['layers'][1]],
                                               n1=train_config['mlp_n1'],
                                               n2=train_config['mlp_n2'],
                                               l1=train_config['mlp_l1_dict']['fcn1/kernel'])
                    fcn1_bias = transfer_mlp(name='bias',
                                             param_prev=fcn1_bias_prev,
                                             param_upd=fcn1_bias_upd,
                                             param_shape=[hyperparams['layers'][1]],
                                             n1=train_config['mlp_n1'],
                                             n2=train_config['mlp_n2'],
                                             l1=train_config['mlp_l1_dict']['fcn1/bias'])
                with tf.variable_scope('fcn2'):
                    fcn2_kernel = transfer_mlp(name='kernel',
                                               param_prev=fcn2_kernel_prev,
                                               param_upd=fcn2_kernel_upd,
                                               param_shape=[hyperparams['layers'][1], hyperparams['layers'][2]],
                                               n1=train_config['mlp_n1'],
                                               n2=train_config['mlp_n2'],
                                               l1=train_config['mlp_l1_dict']['fcn2/kernel'])
                    fcn2_bias = transfer_mlp(name='bias',
                                             param_prev=fcn2_bias_prev,
                                             param_upd=fcn2_bias_upd,
                                             param_shape=[hyperparams['layers'][2]],
                                             n1=train_config['mlp_n1'],
                                             n2=train_config['mlp_n2'],
                                             l1=train_config['mlp_l1_dict']['fcn2/bias'])
                with tf.variable_scope('fcn3'):
                    fcn3_kernel = transfer_mlp(name='kernel',
                                               param_prev=fcn3_kernel_prev,
                                               param_upd=fcn3_kernel_upd,
                                               param_shape=[hyperparams['layers'][2], 1],
                                               n1=train_config['mlp_n1'],
                                               n2=train_config['mlp_n2'],
                                               l1=train_config['mlp_l1_dict']['fcn3/kernel'])
                    fcn3_bias = transfer_mlp(name='bias',
                                             param_prev=fcn3_bias_prev,
                                             param_upd=fcn3_bias_upd,
                                             param_shape=[1],
                                             n1=train_config['mlp_n1'],
                                             n2=train_config['mlp_n2'],
                                             l1=train_config['mlp_l1_dict']['fcn3/bias'])
            # -- transfer mlp end ----

            # -- update op begin -------
            self.fcn1_kernel_upd_op = fcn1_kernel_upd.assign(fcn1_kernel)
            self.fcn1_bias_upd_op = fcn1_bias_upd.assign(fcn1_bias)
            self.fcn2_kernel_upd_op = fcn2_kernel_upd.assign(fcn2_kernel)
            self.fcn2_bias_upd_op = fcn2_bias_upd.assign(fcn2_bias)
            self.fcn3_kernel_upd_op = fcn3_kernel_upd.assign(fcn3_kernel)
            self.fcn3_bias_upd_op = fcn3_bias_upd.assign(fcn3_bias)
            # -- update op end -------

        else:
            # -- create mlp begin ---
            concat_dim = hyperparams['user_embed_dim'] + hyperparams['item_embed_dim'] * 2
            with tf.variable_scope('fcn1'):
                fcn1_kernel = tf.get_variable('kernel', [concat_dim, hyperparams['layers'][1]])
                fcn1_bias = tf.get_variable('bias', [hyperparams['layers'][1]])
            with tf.variable_scope('fcn2'):
                fcn2_kernel = tf.get_variable('kernel', [hyperparams['layers'][1], hyperparams['layers'][2]])
                fcn2_bias = tf.get_variable('bias', [hyperparams['layers'][2]])
            with tf.variable_scope('fcn3'):
                fcn3_kernel = tf.get_variable('kernel', [hyperparams['layers'][2], 1])
                fcn3_bias = tf.get_variable('bias', [1])
            # -- create mlp end ---

        # -- emb begin -------
        u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)  # [B, H]
        i_emb = tf.nn.embedding_lookup(item_emb_w, self.i)  # [B, H]
        h_emb = tf.nn.embedding_lookup(item_emb_w, self.hist_i)  # [B, T, H]
        u_hist = average_pooling(h_emb, self.hist_len)  # [B, H]
        # -- emb end -------

        # -- mlp begin -------
        fcn = tf.concat([u_emb, u_hist, i_emb], axis=-1)  # [B, H x 3]
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
            self.hist_i: batch[2],
            self.hist_len: batch[3],
            self.y: batch[4],
            self.base_lr: self.train_config['base_lr'],
        })
        return loss

    def train_transfer(self, sess, batch):
        loss, _, = sess.run([self.loss, self.train_transfer_op], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[2],
            self.hist_len: batch[3],
            self.y: batch[4],
            self.transfer_lr: self.train_config['transfer_lr'],
        })
        return loss

    def update(self, sess):
        if self.train_config['transfer_emb']:
            sess.run([self.user_emb_w_upd_op,
                      self.item_emb_w_upd_op])
        if self.train_config['transfer_mlp']:
            sess.run([self.fcn1_kernel_upd_op,
                      self.fcn1_bias_upd_op,
                      self.fcn2_kernel_upd_op,
                      self.fcn2_bias_upd_op,
                      self.fcn3_kernel_upd_op,
                      self.fcn3_bias_upd_op])

    def inference(self, sess, batch):
        scores, losses = sess.run([self.scores, self.losses], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[2],
            self.hist_len: batch[3],
            self.y: batch[4],
        })
        return scores, losses
