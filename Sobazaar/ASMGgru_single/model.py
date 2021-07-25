from __future__ import division
from __future__ import print_function
import tensorflow as tf


def average_pooling(emb, seq_len):
    mask = tf.sequence_mask(seq_len, tf.shape(emb)[-2], dtype=tf.float32)  # [B, T]
    mask = tf.expand_dims(mask, -1)  # [B, T, 1]
    emb *= mask  # [B, T, H]
    sum_pool = tf.reduce_sum(emb, -2)  # [B, H]
    avg_pool = tf.div(sum_pool, tf.expand_dims(tf.cast(seq_len, tf.float32), -1) + 1e-8)  # [B, H]
    return avg_pool


def gru_parallel(name, prev_h, cur_x, param_dim, hidden_dim):
    """
    perform one step of gru with tensor parallelization
    :param name: parameter name
    :param prev_h: [N, param_dim, 1, hidden_dim]
    :param cur_x: [N, param_dim, 1, 1]
    :param param_dim: dimension of (flattened) parameter
    :param hidden_dim: dimension of gru hidden state
    :return: cur_h, cur_y
    """
    with tf.variable_scope('gru_' + name, reuse=tf.AUTO_REUSE):
        N = tf.shape(prev_h)[0]

        # gate params
        u_z = tf.tile(tf.expand_dims(tf.get_variable(name='u_z', shape=[param_dim, 1, hidden_dim]), 0), [N, 1, 1, 1])
        w_z = tf.tile(tf.expand_dims(tf.get_variable(name='w_z', shape=[param_dim, hidden_dim, hidden_dim]), 0), [N, 1, 1, 1])
        z = tf.sigmoid(tf.matmul(cur_x, u_z) + tf.matmul(prev_h, w_z))  # [N, param_dim, 1, hidden_dim]

        u_r = tf.tile(tf.expand_dims(tf.get_variable(name='u_r', shape=[param_dim, 1, hidden_dim]), 0), [N, 1, 1, 1])
        w_r = tf.tile(tf.expand_dims(tf.get_variable(name='w_r', shape=[param_dim, hidden_dim, hidden_dim]), 0), [N, 1, 1, 1])
        r = tf.sigmoid(tf.matmul(cur_x, u_r) + tf.matmul(prev_h, w_r))  # [N, param_dim, 1, hidden_dim]

        u_g = tf.tile(tf.expand_dims(tf.get_variable(name='u_g', shape=[param_dim, 1, hidden_dim]), 0), [N, 1, 1, 1])
        w_g = tf.tile(tf.expand_dims(tf.get_variable(name='w_g', shape=[param_dim, hidden_dim, hidden_dim]), 0), [N, 1, 1, 1])
        cur_h_tilde = tf.tanh(tf.matmul(cur_x, u_g) + tf.matmul(prev_h * r, w_g))  # [N, param_dim, 1, hidden_dim]

        cur_h = (1 - z) * prev_h + z * cur_h_tilde  # [N, param_dim, 1, hidden_dim]

        # params to generate cur_y
        w_hy = tf.tile(tf.expand_dims(tf.get_variable(name='w_hy', shape=[param_dim, hidden_dim, 1]), 0), [N, 1, 1, 1])
        b_y = tf.tile(tf.expand_dims(tf.get_variable(name='b_y', shape=[param_dim, 1, 1]), 0), [N, 1, 1, 1])

        # cur_y = tf.tanh(tf.matmul(cur_h, w_hy) + b_y)  # [N, param_dim, 1, 1]
        cur_y = tf.matmul(cur_h, w_hy) + b_y  # [N, param_dim, 1, 1]

    return cur_h, cur_y


def lstm_parallel(name, prev_h, prev_c, cur_x, param_dim, hidden_dim):
    """
    perform one step of lstm with tensor parallelization
    :param name: parameter name
    :param prev_h: [N, param_dim, 1, hidden_dim]
    :param prev_c: [N, param_dim, 1, hidden_dim]
    :param cur_x: [N, param_dim, 1, 1]
    :param param_dim: dimension of (flattened) parameter
    :param hidden_dim: dimension of lstm hidden state
    :return: cur_h, cur_c, cur_y
    """
    with tf.variable_scope('lstm_' + name, reuse=tf.AUTO_REUSE):
        N = tf.shape(prev_h)[0]

        # gate params
        u_i = tf.tile(tf.expand_dims(tf.get_variable(name='u_i', shape=[param_dim, 1, hidden_dim]), 0), [N, 1, 1, 1])
        w_i = tf.tile(tf.expand_dims(tf.get_variable(name='w_i', shape=[param_dim, hidden_dim, hidden_dim]), 0), [N, 1, 1, 1])
        i = tf.sigmoid(tf.matmul(cur_x, u_i) + tf.matmul(prev_h, w_i))  # [N, param_dim, 1, hidden_dim]

        u_f = tf.tile(tf.expand_dims(tf.get_variable(name='u_f', shape=[param_dim, 1, hidden_dim]), 0), [N, 1, 1, 1])
        w_f = tf.tile(tf.expand_dims(tf.get_variable(name='w_f', shape=[param_dim, hidden_dim, hidden_dim]), 0), [N, 1, 1, 1])
        f = tf.sigmoid(tf.matmul(cur_x, u_f) + tf.matmul(prev_h, w_f))  # [N, param_dim, 1, hidden_dim]

        u_o = tf.tile(tf.expand_dims(tf.get_variable(name='u_o', shape=[param_dim, 1, hidden_dim]), 0), [N, 1, 1, 1])
        w_o = tf.tile(tf.expand_dims(tf.get_variable(name='w_o', shape=[param_dim, hidden_dim, hidden_dim]), 0), [N, 1, 1, 1])
        o = tf.sigmoid(tf.matmul(cur_x, u_o) + tf.matmul(prev_h, w_o))  # [N, param_dim, 1, hidden_dim]

        u_g = tf.tile(tf.expand_dims(tf.get_variable(name='u_g', shape=[param_dim, 1, hidden_dim]), 0), [N, 1, 1, 1])
        w_g = tf.tile(tf.expand_dims(tf.get_variable(name='w_g', shape=[param_dim, hidden_dim, hidden_dim]), 0), [N, 1, 1, 1])
        cur_c_tilde = tf.tanh(tf.matmul(cur_x, u_g) + tf.matmul(prev_h, w_g))  # [N, param_dim, 1, hidden_dim]

        cur_c = tf.sigmoid(f * prev_c + i * cur_c_tilde)  # [N, param_dim, 1, hidden_dim]
        cur_h = tf.tanh(cur_c) * o  # [N, param_dim, 1, hidden_dim]

        # params to generate cur_y
        w_hy = tf.tile(tf.expand_dims(tf.get_variable(name='w_hy', shape=[param_dim, hidden_dim, 1]), 0), [N, 1, 1, 1])
        b_y = tf.tile(tf.expand_dims(tf.get_variable(name='b_y', shape=[param_dim, 1, 1]), 0), [N, 1, 1, 1])

        # cur_y = tf.tanh(tf.matmul(cur_h, w_hy) + b_y)  # [N, param_dim, 1, 1]
        cur_y = tf.matmul(cur_h, w_hy) + b_y  # [N, param_dim, 1, 1]

    return cur_h, cur_c, cur_y


def vanilla_parallel(name, prev_h, cur_x, param_dim, hidden_dim):
    """
    perform one step of vanilla rnn with tensor parallelization
    :param name: parameter name
    :param prev_h: [N, param_dim, 1, hidden_dim]
    :param cur_x: [N, param_dim, 1, 1]
    :param param_dim: dimension of (flattened) parameter
    :param hidden_dim: dimension of vanilla rnn hidden state
    :return: cur_h, cur_y
    """
    with tf.variable_scope('vanilla_' + name, reuse=tf.AUTO_REUSE):
        N = tf.shape(prev_h)[0]

        # params to generate cur_h
        w_hh = tf.tile(tf.expand_dims(tf.get_variable(name='w_hh', shape=[param_dim, hidden_dim, hidden_dim]), 0), [N, 1, 1, 1])
        w_xh = tf.tile(tf.expand_dims(tf.get_variable(name='w_xh', shape=[param_dim, 1, hidden_dim]), 0), [N, 1, 1, 1])
        b_h = tf.tile(tf.expand_dims(tf.get_variable(name='w_xh', shape=[param_dim, 1, hidden_dim]), 0), [N, 1, 1, 1])

        cur_h = tf.tanh(tf.matmul(prev_h, w_hh) + tf.matmul(cur_x, w_xh) + b_h)  # [N, param_dim, 1, hidden_dim]

        # params to generate cur_y
        w_hy = tf.tile(tf.expand_dims(tf.get_variable(name='w_hy', shape=[param_dim, hidden_dim, 1]), 0), [N, 1, 1, 1])
        b_y = tf.tile(tf.expand_dims(tf.get_variable(name='b_y', shape=[param_dim, 1, 1]), 0), [N, 1, 1, 1])

        # cur_y = tf.tanh(tf.matmul(cur_h, w_hy) + b_y)  # [N, param_dim, 1, 1]
        cur_y = tf.matmul(cur_h, w_hy) + b_y  # [N, param_dim, 1, 1]

    return cur_h, cur_y


rnn_dict = {'vanilla': vanilla_parallel,
            'gru': gru_parallel,
            'lstm': lstm_parallel}


def rnn_combine(name, param_ls, param_shape, seq_length, rnn_type, init_h, hidden_dim):
    """
    perform rnn on sequence of parameters and output the final hs and cur_y
    :param name: param name
    :param param_ls: param list of seq_length
    :param param_shape: param shape [d1, d2, ...]
    :param seq_length: input sequence length
    :param rnn_type: type of rnn cell
    :param init_h: initial h [d1, d2, ..., hidden_dim]
    :param hidden_dim: rnn hidden dimensions
    :return: final hs [d1, d2, ..., hidden_dim, seq_length], final cur_y [d1, d2, ...]
    """
    with tf.variable_scope('rnn_combine'):
        if 'fcn' in name:
            # N =1, param_dim = flat_dim
            if 'bias' in name:
                param_dim = param_shape[0]  # d1
            else:  # 'kernel'
                param_dim = param_shape[0] * param_shape[1]  # d1 x d2
            flat_param_ls = tf.reshape(param_ls, [-1, seq_length])  # [param_dim, seq_length]
            prev_h = tf.expand_dims(tf.expand_dims(tf.reshape(init_h, [-1, hidden_dim]), 1), 0)  # [1, param_dim, 1, hidden_dim]
            h_ls = []
            for i in range(seq_length):
                cur_x = flat_param_ls[:, i]  # [param_dim]
                cur_x = tf.expand_dims(tf.expand_dims(tf.expand_dims(cur_x, -1), -1), 0)  # [1, param_dim, 1, 1]
                cur_h, cur_y = rnn_dict[rnn_type](name, prev_h, cur_x, param_dim, hidden_dim)  # [1, param_dim, 1, hidden_dim], [1, param_dim, 1, 1]
                prev_h = cur_h
                h_ls.append(tf.reshape(cur_h, [param_dim, hidden_dim]))  # [param_dim, hidden_dim]
            hs = tf.stack(h_ls, -1)  # [param_dim, hidden_dim, seq_length]
            hs = tf.reshape(hs, param_shape + [hidden_dim, seq_length])  # [d1, d2, hidden_dim, seq_length] / [d1, hidden_dim, seq_length]
            cur_y = tf.reshape(cur_y, param_shape)  # [d1, d2] / [d1]
        else:  # 'emb' in name
            # N = num, param_dim = embed_dim
            param_dim = param_shape[1]
            prev_h = tf.expand_dims(init_h, 2)  # [N, param_dim, 1, hidden_dim]
            h_ls = []
            for i in range(seq_length):
                cur_x = param_ls[:, :, i]  # [N, param_dim]
                cur_x = tf.expand_dims(tf.expand_dims(cur_x, -1), -1)  # [N, param_dim, 1, 1]
                cur_h, cur_y = rnn_dict[rnn_type](name, prev_h, cur_x, param_dim, hidden_dim)  # [N, param_dim, 1, hidden_dim], [N, param_dim, 1, 1]
                prev_h = cur_h
                h_ls.append(tf.reshape(cur_h, param_shape + [hidden_dim]))  # [N, param_dim, hidden_dim]
            hs = tf.stack(h_ls, -1)  # [num, embed_dim, hidden_dim, seq_length]
            cur_y = tf.reshape(cur_y, param_shape)  # [num, embed_dim]
    return hs, cur_y


class ASMGrnnSingle(object):

    def __init__(self, hyperparams, emb_ls_dict, mlp_ls_dict, init_h_dict, train_config=None):

        self.train_config = train_config

        # create placeholder
        self.u = tf.placeholder(tf.int32, [None])  # [B]
        self.i = tf.placeholder(tf.int32, [None])  # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.hist_len = tf.placeholder(tf.int32, [None])  # [B]
        self.y = tf.placeholder(tf.float32, [None])  # [B]
        self.meta_lr = tf.placeholder(tf.float32, [], name='meta_lr')  # scalar

        # -- create emb_w_ls begin ----
        user_emb_w_ls = tf.convert_to_tensor(emb_ls_dict['user_emb_w'], tf.float32)  # [num_users, user_embed_dim, seq_length]
        item_emb_w_ls = tf.convert_to_tensor(emb_ls_dict['item_emb_w'], tf.float32)  # [num_items, item_embed_dim, seq_length]
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
            user_emb_w_init_h = tf.convert_to_tensor(init_h_dict['user_emb_w'], tf.float32)  # [num_users, user_embed_dim, emb_hidden_dim]
            self.user_emb_w_hs, user_emb_w = rnn_combine(name='user_emb_w',
                                                         param_ls=user_emb_w_ls,
                                                         param_shape=[hyperparams['num_users'], hyperparams['user_embed_dim']],
                                                         seq_length=train_config['seq_length'],
                                                         rnn_type=train_config['rnn_type'],
                                                         init_h=user_emb_w_init_h,
                                                         hidden_dim=train_config['emb_hidden_dim'])
            item_emb_w_init_h = tf.convert_to_tensor(init_h_dict['item_emb_w'], tf.float32)  # [num_items, item_embed_dim, emb_hidden_dim]
            self.item_emb_w_hs, item_emb_w = rnn_combine(name='item_emb_w',
                                                         param_ls=item_emb_w_ls,
                                                         param_shape=[hyperparams['num_items'], hyperparams['item_embed_dim']],
                                                         seq_length=train_config['seq_length'],
                                                         rnn_type=train_config['rnn_type'],
                                                         init_h=item_emb_w_init_h,
                                                         hidden_dim=train_config['emb_hidden_dim'])
        # -- generate emb_w end ----

        # -- generate mlp begin ----
        with tf.variable_scope('meta_mlp'):
            concat_dim = hyperparams['user_embed_dim'] + hyperparams['item_embed_dim'] * 2
            fcn1_kernel_init_h = tf.convert_to_tensor(init_h_dict['fcn1_kernel'], tf.float32)  # [concat_dim, l1, mlp_hidden_dim]
            self.fcn1_kernel_hs, fcn1_kernel = rnn_combine(name='fcn1_kernel',
                                                           param_ls=fcn1_kernel_ls,
                                                           param_shape=[concat_dim, hyperparams['layers'][1]],
                                                           seq_length=train_config['seq_length'],
                                                           rnn_type=train_config['rnn_type'],
                                                           init_h=fcn1_kernel_init_h,
                                                           hidden_dim=train_config['mlp_hidden_dim'])
            fcn1_bias_init_h = tf.convert_to_tensor(init_h_dict['fcn1_bias'], tf.float32)  # [l1, mlp_hidden_dim]
            self.fcn1_bias_hs, fcn1_bias = rnn_combine(name='fcn1_bias',
                                                       param_ls=fcn1_bias_ls,
                                                       param_shape=[hyperparams['layers'][1]],
                                                       seq_length=train_config['seq_length'],
                                                       rnn_type=train_config['rnn_type'],
                                                       init_h=fcn1_bias_init_h,
                                                       hidden_dim=train_config['mlp_hidden_dim'])
            fcn2_kernel_init_h = tf.convert_to_tensor(init_h_dict['fcn2_kernel'], tf.float32)  # [l1, l2, mlp_hidden_dim]
            self.fcn2_kernel_hs, fcn2_kernel = rnn_combine(name='fcn2_kernel',
                                                           param_ls=fcn2_kernel_ls,
                                                           param_shape=[hyperparams['layers'][1], hyperparams['layers'][2]],
                                                           seq_length=train_config['seq_length'],
                                                           rnn_type=train_config['rnn_type'],
                                                           init_h=fcn2_kernel_init_h,
                                                           hidden_dim=train_config['mlp_hidden_dim'])
            fcn2_bias_init_h = tf.convert_to_tensor(init_h_dict['fcn2_bias'], tf.float32)  # [l2, mlp_hidden_dim]
            self.fcn2_bias_hs, fcn2_bias = rnn_combine(name='fcn2_bias',
                                                       param_ls=fcn2_bias_ls,
                                                       param_shape=[hyperparams['layers'][2]],
                                                       seq_length=train_config['seq_length'],
                                                       rnn_type=train_config['rnn_type'],
                                                       init_h=fcn2_bias_init_h,
                                                       hidden_dim=train_config['mlp_hidden_dim'])
            fcn3_kernel_init_h = tf.convert_to_tensor(init_h_dict['fcn3_kernel'], tf.float32)  # [l2, 1, mlp_hidden_dim]
            self.fcn3_kernel_hs, fcn3_kernel = rnn_combine(name='fcn3_kernel',
                                                           param_ls=fcn3_kernel_ls,
                                                           param_shape=[hyperparams['layers'][2], 1],
                                                           seq_length=train_config['seq_length'],
                                                           rnn_type=train_config['rnn_type'],
                                                           init_h=fcn3_kernel_init_h,
                                                           hidden_dim=train_config['mlp_hidden_dim'])
            fcn3_bias_init_h = tf.convert_to_tensor(init_h_dict['fcn3_bias'], tf.float32)  # [1, mlp_hidden_dim]
            self.fcn3_bias_hs, fcn3_bias = rnn_combine(name='fcn3_bias',
                                                       param_ls=fcn3_bias_ls,
                                                       param_shape=[1],
                                                       seq_length=train_config['seq_length'],
                                                       rnn_type=train_config['rnn_type'],
                                                       init_h=fcn3_bias_init_h,
                                                       hidden_dim=train_config['mlp_hidden_dim'])
        # -- generate mlp end ----

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

    def get_h_dict(self, sess):
        user_emb_w_hs, item_emb_w_hs, \
        fcn1_kernel_hs, fcn1_bias_hs, fcn2_kernel_hs, fcn2_bias_hs, \
        fcn3_kernel_hs, fcn3_bias_hs = sess.run([
            self.user_emb_w_hs, self.item_emb_w_hs,
            self.fcn1_kernel_hs, self.fcn1_bias_hs, self.fcn2_kernel_hs, self.fcn2_bias_hs,
            self.fcn3_kernel_hs, self.fcn3_bias_hs])
        h_dict = {'user_emb_w': user_emb_w_hs,
                  'item_emb_w': item_emb_w_hs,
                  'fcn1_kernel': fcn1_kernel_hs,
                  'fcn1_bias': fcn1_bias_hs,
                  'fcn2_kernel': fcn2_kernel_hs,
                  'fcn2_bias': fcn2_bias_hs,
                  'fcn3_kernel': fcn3_kernel_hs,
                  'fcn3_bias': fcn3_bias_hs}
        return h_dict

    def train_meta(self, sess, batch):
        loss, _, = sess.run([self.loss, self.train_meta_op], feed_dict={
            self.u: batch[0],
            self.i: batch[1],
            self.hist_i: batch[2],
            self.hist_len: batch[3],
            self.y: batch[4],
            self.meta_lr: self.train_config['meta_lr'],
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
