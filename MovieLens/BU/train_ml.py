from __future__ import division
from __future__ import print_function
import os
from engine import *
from model import *
from utils import *

np.random.seed(1234)
tf.set_random_seed(123)

# load data to df
start_time = time.time()

data_df = pd.read_csv('../../datasets/ml_5yr_2014_2018_30seq.csv')
meta_df = pd.read_csv('../../datasets/ml_5yr_2014_2018_30seq_item_meta.csv')

data_df['itemSeq'] = data_df['itemSeq'].fillna('')  # empty seq are NaN
data_df['itemSeq'] = data_df['itemSeq'].apply(lambda x: [int(item) for item in x.split('#') if item != ''])
meta_df['cateId'] = meta_df['cateId'].apply(lambda x: [int(cate) for cate in x.split('#') if cate != ''])
meta_df = meta_df.sort_values(['itemId'], ascending=True).reset_index(drop=True)
cate_ls = meta_df['cateId'].tolist()

print('Done loading data! time elapsed: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

num_users = data_df['userId'].max() + 1
num_items = data_df['itemId'].max() + 1
num_cates = max([max(i) for i in cate_ls]) + 1
cates, cate_lens = process_cate(cate_ls)

train_config = {'method': 'BU_by_period',
                'dir_name': 'BU_train11-23_test24-30_7_1epoch',  # edit train test period, window size, number of epochs
                'pretrain_model': 'pretrain_train1-10_test11_10epoch_0.001',
                'start_date': 20140101,  # overall train start date
                'end_date': 20181231,  # overall train end date
                'num_periods': 31,  # number of periods divided into
                'train_start_period': 11,
                'test_start_period': 24,
                'window_size': 7,  # number of periods or 'full' for full retraining
                'cur_periods': None,  # current batch periods
                'next_period': None,  # next incremental period
                'cur_set_size': None,  # current batch dataset size
                'next_set_size': None,  # next incremental dataset size
                'period_alias': None,  # individual period directory alias to save ckpts
                'restored_ckpt_mode': 'best auc',  # mode to search the checkpoint to restore, 'best auc', 'best gauc', 'last'
                'restored_ckpt': None,  # configure in the for loop

                'base_optimizer': 'adam',  # base model optimizer: adam, rmsprop, sgd
                'base_lr': None,  # base model learning rate
                'base_bs': 1024,  # base model batch size
                'base_num_epochs': 1,  # base model number of epochs
                'shuffle': True,  # whether to shuffle the dataset for each epoch
                }

EmbMLP_hyperparams = {'num_users': num_users,
                      'num_items': num_items,
                      'num_cates': num_cates,
                      'user_embed_dim': 8,
                      'item_embed_dim': 8,
                      'cate_embed_dim': 8,
                      'layers': [40, 20, 10, 1]
                      }

# sort train data into periods based on num_periods
data_df = data_df[(data_df['date'] >= train_config['start_date']) & (data_df['date'] <= train_config['end_date'])]
data_df = data_df.sort_values(['timestamp']).reset_index(drop=True)
records_per_period = int(len(data_df) / train_config['num_periods'])
data_df['index'] = data_df.index
data_df['period'] = data_df['index'].apply(lambda x: int(x / records_per_period) + 1)
data_df = data_df[data_df.period != train_config['num_periods'] + 1]  # delete last extra period
period_df = data_df.groupby('period')['date'].agg(['count', 'min', 'max'])
data_df = data_df.drop(['index', 'date', 'timestamp'], axis=1)


def train_base():

    # create an engine instance with base_model
    engine = Engine(sess, base_model)

    train_start_time = time.time()

    max_auc = 0
    best_logloss = 0

    for epoch_id in range(1, train_config['base_num_epochs'] + 1):

        print('Training Base Model Epoch {} Start!'.format(epoch_id))

        base_loss_cur_avg = engine.base_train_an_epoch(epoch_id, cur_set, train_config)
        print('Epoch {} Done! time elapsed: {}, base_loss_cur_avg {:.4f}'.format(
            epoch_id,
            time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time)),
            base_loss_cur_avg))

        cur_auc, cur_logloss = engine.test(cur_set, train_config)
        next_auc, next_logloss = engine.test(next_set, train_config)
        print('cur_auc {:.4f}, cur_logloss {:.4f}, next_auc {:.4f}, next_logloss {:.4f}'.format(
            cur_auc,
            cur_logloss,
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

        if next_auc > max_auc:
            max_auc = next_auc
            best_logloss = next_logloss

    if i >= train_config['test_start_period']:
        test_aucs.append(max_auc)
        test_loglosses.append(best_logloss)


orig_dir_name = train_config['dir_name']

for base_lr in [1e-3]:

    print('')
    print('base_lr', base_lr)

    train_config['base_lr'] = base_lr

    train_config['dir_name'] = orig_dir_name + '_' + str(base_lr)
    print('dir_name: ', train_config['dir_name'])

    test_aucs = []
    test_loglosses = []

    for i in range(train_config['train_start_period'], train_config['num_periods']):

        # configure cur_periods, next_period
        if train_config['window_size'] == 'full':
            train_config['cur_periods'] = [i - prev_num for prev_num in reversed(range(i))]
        else:
            train_config['cur_periods'] = [i - prev_num for prev_num in reversed(range(train_config['window_size']))]
        train_config['next_period'] = i + 1
        print('')
        print('current periods: {}, next period: {}'.format(
            train_config['cur_periods'],
            train_config['next_period']))
        print('')

        # create current and next set
        cur_set = data_df[data_df['period'].isin(train_config['cur_periods'])]
        next_set = data_df[data_df['period'] == train_config['next_period']]
        train_config['cur_set_size'] = len(cur_set)
        train_config['next_set_size'] = len(next_set)
        print('current set size', len(cur_set), 'next set size', len(next_set))

        train_config['period_alias'] = 'period' + str(i)

        # checkpoints directory
        ckpts_dir = os.path.join('ckpts', train_config['dir_name'], train_config['period_alias'])
        if not os.path.exists(ckpts_dir):
            os.makedirs(ckpts_dir)

        if i == train_config['train_start_period']:
            search_alias = os.path.join('../pretrain/ckpts', train_config['pretrain_model'], 'Epoch*')
            train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
        else:
            prev_period_alias = 'period' + str(i - 1)
            search_alias = os.path.join('ckpts', train_config['dir_name'], prev_period_alias, 'Epoch*')
            train_config['restored_ckpt'] = search_ckpt(search_alias, mode=train_config['restored_ckpt_mode'])
        print('restored checkpoint: {}'.format(train_config['restored_ckpt']))

        # write train_config to text file
        with open(os.path.join(ckpts_dir, 'config.txt'), mode='w') as f:
            f.write('train_config: ' + str(train_config) + '\n')
            f.write('\n')
            f.write('EmbMLP_hyperparams: ' + str(EmbMLP_hyperparams) + '\n')

        # build base model computation graph
        tf.reset_default_graph()
        base_model = EmbMLP(cates, cate_lens, EmbMLP_hyperparams, train_config=train_config)

        # create session
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, train_config['restored_ckpt'])
            train_base()

        if i >= train_config['test_start_period']:
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
