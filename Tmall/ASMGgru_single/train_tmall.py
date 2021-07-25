from __future__ import division
from __future__ import print_function
import os
import pickle
from engine import *
from model import *
from utils import *

np.random.seed(1234)
tf.set_random_seed(123)

# load data to df
start_time = time.time()

data_df = pd.read_csv('../../datasets/tmall_1mth_2014_item20user50k_1neg_30seq.csv')
meta_df = pd.read_csv('../../datasets/tmall_1mth_2014_item20user50k_1neg_30seq_item_meta.csv')

data_df['itemSeq'] = data_df['itemSeq'].fillna('')  # empty seq are NaN
data_df['itemSeq'] = data_df['itemSeq'].apply(lambda x: [int(item) for item in x.split('#') if item != ''])
meta_df['cateId'] = meta_df['cateId'].apply(lambda x: [x])
meta_df = meta_df.sort_values(['itemId'], ascending=True).reset_index(drop=True)
cate_ls = meta_df['cateId'].tolist()

print('Done loading data! time elapsed: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

num_users = data_df['userId'].max() + 1
num_items = data_df['itemId'].max() + 1
num_cates = max([max(i) for i in cate_ls]) + 1
cates, cate_lens = process_cate(cate_ls)

train_config = {'method': 'ASMGgru_single_by_date',
                'dir_name': 'ASMGgru_single_train11-23_test24-30_4emb_4mlp_1epoch',  # edit train test period, rnn hidden size for emb and mlp, number of epochs
                'niu_dir_name': 'NIU_train11-23_test24-30_1epoch_0.01',  # input model sequence directory
                'train_start_date': 20141011,
                'train_end_date': 20141031,
                'test_start_date': 20141024,
                'cur_date': None,  # current incremental date
                'next_date': None,  # next incremental date
                'next_set_size': None,  # next incremental dataset size
                'date_alias': None,  # individual date directory alias to save ckpts
                'restored_ckpt_mode': 'best auc',  # mode to search the ckpt to restore: 'best auc', 'best logloss', 'last'
                'restored_ckpt': None,  # restored meta generator checkpoint

                'seq_length': None,   # length of input model sequence
                'rnn_type': 'gru',  # type of rnn cell: vanilla, gru
                'emb_hidden_dim': 4,  # rnn hidden size for embedding layers parameters
                'mlp_hidden_dim': 4,  # rnn hidden size for MLP layers parameters
                'test_stop_train': False,  # whether to stop updating meta generator during test periods

                'meta_optimizer': 'adam',  # meta generator optimizer: adam, rmsprop, sgd
                'meta_lr': None,  # meta generator learning rate
                'meta_bs': 1024,  # meta generator batch size
                'meta_num_epochs': 1,  # meta generator number of epochs
                'shuffle': True,  # whether to shuffle the dataset for each epoch
                }

EmbMLP_hyperparams = {'num_users': num_users,
                      'num_items': num_items,
                      'num_cates': num_cates,
                      'user_embed_dim': 8,
                      'item_embed_dim': 8,
                      'cate_embed_dim': 8,
                      'layers': [40, 20, 10, 1]  # input dim is user_embed_dim + (item_embed_dim + cate_embed_dim) x 2
                      }


def collect_params():
    """
    collect list of parameters for input model sequence
    :return: emb_ls_dict, mlp_ls_dict
    """

    collect_params_start_time = time.time()

    emb_ls = ['user_emb_w', 'item_emb_w', 'cate_emb_w']
    mlp_ls = ['fcn1/kernel', 'fcn2/kernel', 'fcn3/kernel', 'fcn3/bias', 'fcn1/bias', 'fcn2/bias']

    # collect input model sequence from niu_dir
    emb_dict_ls = []
    mlp_dict_ls = []
    for prev_num in reversed(range(train_config['seq_length'])):
        date_alias = 'date' + str(i - prev_num)
        alias = os.path.join('../IU/ckpts', train_config['niu_dir_name'], date_alias, 'Epoch*')
        restored_ckpt = search_ckpt(alias, mode=train_config['restored_ckpt_mode'])
        print('restored model {}: {}'.format(i - prev_num, restored_ckpt))
        emb_dict = {name: tf.train.load_checkpoint(restored_ckpt).get_tensor(name)
                    for name, _ in tf.train.list_variables(restored_ckpt) if name in emb_ls}
        mlp_dict = {name: tf.train.load_checkpoint(restored_ckpt).get_tensor(name)
                    for name, _ in tf.train.list_variables(restored_ckpt) if name in mlp_ls}
        emb_dict_ls.append(emb_dict)
        mlp_dict_ls.append(mlp_dict)

    # concat sequence for different parameters on the last axis
    emb_ls_dict_ = {}
    for k in emb_dict_ls[0].keys():
        for emb_dict in emb_dict_ls:
            if k not in emb_ls_dict_.keys():
                emb_ls_dict_[k] = np.expand_dims(emb_dict[k], axis=-1)
            else:
                emb_ls_dict_[k] = np.concatenate((emb_ls_dict_[k], np.expand_dims(emb_dict[k], axis=-1)), axis=-1)

    mlp_ls_dict_ = {}
    for k in mlp_dict_ls[0].keys():
        for mlp_dict in mlp_dict_ls:
            if k not in mlp_ls_dict_.keys():
                mlp_ls_dict_[k] = np.expand_dims(mlp_dict[k], axis=-1)
            else:
                mlp_ls_dict_[k] = np.concatenate((mlp_ls_dict_[k], np.expand_dims(mlp_dict[k], axis=-1)), axis=-1)

    # check that the shapes are correct
    for k in emb_ls_dict_.keys():
        print(k, np.shape(emb_ls_dict_[k]))
    for k in mlp_ls_dict_.keys():
        print(k, np.shape(mlp_ls_dict_[k]))

    print('collect params time elapsed: {}'.format(
        time.strftime('%H:%M:%S', time.gmtime(time.time() - collect_params_start_time))))

    return emb_ls_dict_, mlp_ls_dict_


def collect_init_h():
    """
    collect previously trained initial hidden state
    :return: init_h_dict
    """
    collect_init_h_start_time = time.time()
    path = os.path.join('ckpts', train_config['dir_name'], prev_date_alias, 'h_dict.pkl')
    with open(path, mode='r') as fh:
        h_dict = pickle.load(fh)
    init_h_dict_ = {}
    for k in h_dict.keys():
        init_h_dict_[k] = h_dict[k][..., 0]
    print('collect init h time elapsed: {}'.format(
        time.strftime('%H:%M:%S', time.gmtime(time.time() - collect_init_h_start_time))))

    return init_h_dict_


def test_and_train_meta():

    # create an engine instance with asmg_model
    engine = Engine(sess, asmg_model)

    test_start_time = time.time()
    print('Testing Meta Generator Start!')
    next_auc, next_logloss = engine.test(next_set, train_config)
    print('Done! time elapsed: {}, next_auc {:.4f}, next_logloss {:.4f}'.format(
        time.strftime('%H:%M:%S', time.gmtime(time.time() - test_start_time)),
        next_auc,
        next_logloss))
    print('')

    if i >= train_config['test_start_date']:
        test_aucs.append(next_auc)
        test_loglosses.append(next_logloss)

    if i < train_config['test_start_date'] or not train_config['test_stop_train']:

        train_start_time = time.time()

        for epoch_id in range(1, train_config['meta_num_epochs'] + 1):

            print('Training Meta Generator Epoch {} Start!'.format(epoch_id))

            meta_loss_next_avg = engine.meta_train_an_epoch(epoch_id, next_set, train_config)
            print('Epoch {} Done! time elapsed: {}, meta_loss_next_avg {:.4f}'.format(
                epoch_id,
                time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time)),
                meta_loss_next_avg
            ))

            next_auc, next_logloss = engine.test(next_set, train_config)
            print('next_auc {:.4f}, next_logloss {:.4f}'.format(
                next_auc,
                next_logloss))
            print('time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))

            print('')

            # save checkpoint
            checkpoint_alias = 'Epoch{}_TestAUC{:.4f}_TestLOGLOSS{:.4f}.ckpt'.format(
                epoch_id,
                next_auc,
                next_logloss)
            checkpoint_path = os.path.join(ckpts_dir, checkpoint_alias)
            saver.save(sess, checkpoint_path)

        # save h_dict
        h_dict_ = asmg_model.get_h_dict(sess)
        with open(os.path.join(ckpts_dir, 'h_dict.pkl'), mode='w') as fh:
            pickle.dump(h_dict_, fh)

    else:
        # save checkpoint
        checkpoint_alias = 'EpochNA_TestAUC{:.4f}_TestLOGLOSS{:.4f}.ckpt'.format(
            next_auc,
            next_logloss)
        checkpoint_path = os.path.join(ckpts_dir, checkpoint_alias)
        saver.save(sess, checkpoint_path)

        # save h_dict
        h_dict_ = asmg_model.get_h_dict(sess)
        with open(os.path.join(ckpts_dir, 'h_dict.pkl'), mode='w') as fh:
            pickle.dump(h_dict_, fh)


orig_dir_name = train_config['dir_name']

for seq_length in [3]:

    for meta_lr in [1e-2]:

        print('')
        print('seq_length', seq_length, 'meta_lr', meta_lr)

        train_config['seq_length'] = seq_length
        train_config['meta_lr'] = meta_lr

        train_config['dir_name'] = orig_dir_name + '_' + str(seq_length) + '_' + str(meta_lr)
        print('dir_name: ', train_config['dir_name'])

        test_aucs = []
        test_loglosses = []

        start_date = train_config['train_start_date'] + seq_length - 1

        for i in range(start_date, train_config['train_end_date']):

            # configure cur_date, next_date
            train_config['cur_date'] = i
            train_config['next_date'] = i + 1
            print('')
            print('current date: {}, next date: {}'.format(
                train_config['cur_date'],
                train_config['next_date']))
            print('')

            # create next set
            next_set = data_df[data_df['date'] == train_config['next_date']]
            train_config['next_set_size'] = len(next_set)
            print('next set size', len(next_set))

            train_config['date_alias'] = 'date' + str(i)

            # checkpoints directory
            ckpts_dir = os.path.join('ckpts', train_config['dir_name'], train_config['date_alias'])
            if not os.path.exists(ckpts_dir):
                os.makedirs(ckpts_dir)

            if i == start_date:
                train_config['restored_ckpt'] = None
            else:
                prev_date_alias = 'date' + str(i - 1)
                search_alias = os.path.join('ckpts', train_config['dir_name'], prev_date_alias, 'Epoch*')
                train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
            print('restored checkpoint: {}'.format(train_config['restored_ckpt']))

            # write train_config to text file
            with open(os.path.join(ckpts_dir, 'config.txt'), mode='w') as f:
                f.write('train_config: ' + str(train_config) + '\n')
                f.write('\n')
                f.write('EmbMLP_hyperparams: ' + str(EmbMLP_hyperparams) + '\n')

            # collect list of parameters for input model sequence
            emb_ls_dict, mlp_ls_dict = collect_params()

            # collect previously trained initial hidden state
            if i == start_date:
                init_h_dict = {'user_emb_w': np.zeros((EmbMLP_hyperparams['num_users'],
                                                       EmbMLP_hyperparams['user_embed_dim'],
                                                       train_config['emb_hidden_dim'])),
                               'item_emb_w': np.zeros((EmbMLP_hyperparams['num_items'],
                                                       EmbMLP_hyperparams['item_embed_dim'],
                                                       train_config['emb_hidden_dim'])),
                               'cate_emb_w': np.zeros((EmbMLP_hyperparams['num_cates'],
                                                       EmbMLP_hyperparams['cate_embed_dim'],
                                                       train_config['emb_hidden_dim'])),
                               'fcn1_kernel': np.zeros((EmbMLP_hyperparams['layers'][0],
                                                        EmbMLP_hyperparams['layers'][1],
                                                        train_config['mlp_hidden_dim'])),
                               'fcn1_bias': np.zeros((EmbMLP_hyperparams['layers'][1],
                                                      train_config['mlp_hidden_dim'])),
                               'fcn2_kernel': np.zeros((EmbMLP_hyperparams['layers'][1],
                                                        EmbMLP_hyperparams['layers'][2],
                                                        train_config['mlp_hidden_dim'])),
                               'fcn2_bias': np.zeros((EmbMLP_hyperparams['layers'][2],
                                                      train_config['mlp_hidden_dim'])),
                               'fcn3_kernel': np.zeros((EmbMLP_hyperparams['layers'][2],
                                                        EmbMLP_hyperparams['layers'][3],
                                                        train_config['mlp_hidden_dim'])),
                               'fcn3_bias': np.zeros((EmbMLP_hyperparams['layers'][3],
                                                      train_config['mlp_hidden_dim']))}
            else:
                init_h_dict = collect_init_h()

            # build asmg model computation graph
            tf.reset_default_graph()
            asmg_model = ASMGrnnSingle(cates, cate_lens, EmbMLP_hyperparams, emb_ls_dict, mlp_ls_dict, init_h_dict, train_config=train_config)

            # create session
            with tf.Session() as sess:

                # restore meta generator
                if i == start_date:
                    # print([var.name for var in tf.global_variables()])  # check graph variables
                    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                else:
                    restorer = tf.train.Saver()
                    restorer.restore(sess, train_config['restored_ckpt'])
                saver = tf.train.Saver()

                # test and then train meta generator
                test_and_train_meta()

            if i >= train_config['test_start_date']:
                average_auc = sum(test_aucs) / len(test_aucs)
                average_logloss = sum(test_loglosses) / len(test_loglosses)
                print('test aucs', test_aucs)
                print('average auc', average_auc)
                print('')
                print('test loglosses', test_loglosses)
                print('average logloss', average_logloss)

                # write metrics to text file
                with open(os.path.join(ckpts_dir, 'test_metrics.txt'), mode='w') as f:
                    f.write('test_aucs: ' + str(test_aucs) + '\n')
                    f.write('average_auc: ' + str(average_auc) + '\n')
                    f.write('test_loglosses: ' + str(test_loglosses) + '\n')
                    f.write('average_logloss: ' + str(average_logloss) + '\n')
