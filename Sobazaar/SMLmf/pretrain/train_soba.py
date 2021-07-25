from __future__ import division
from __future__ import print_function
import os
import numpy as np
from engine import *
from model import *
from utils import *

np.random.seed(1234)
tf.set_random_seed(123)

# load data to df
start_time = time.time()

data_df = pd.read_csv('../../../datasets/soba_4mth_2014_1neg_30seq.csv')

print('Done loading data! time elapsed: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

num_users = data_df['userId'].max() + 1
num_items = data_df['itemId'].max() + 1

train_config = {'method': 'pretrain',
                'dir_name': 'pretrain_train1-10_test11_10epoch',  # edit train test period range, number of epochs
                'start_date': 20140901,  # overall train start date
                'end_date': 20141231,  # overall train end date
                'num_periods': 31,  # number of periods divided into
                'train_start_period': 1,
                'train_end_period': 10,
                'test_period': 11,
                'train_set_size': None,
                'test_set_size': None,

                'base_optimizer': 'adam',  # base model optimizer: adam, rmsprop, sgd
                'base_lr': None,  # base model learning rate
                'base_bs': 256,  # base model batch size
                'base_num_epochs': 10,  # base model number of epochs
                'shuffle': True,  # whether to shuffle the dataset for each epoch
                }

MF_hyperparams = {'num_users': num_users,
                  'num_items': num_items,
                  'user_embed_dim': 8,
                  'item_embed_dim': 8,
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

    # create an engine instance
    engine = Engine(sess, base_model)

    train_start_time = time.time()

    for epoch_id in range(1, train_config['base_num_epochs'] + 1):

        print('Training Base Model Epoch {} Start!'.format(epoch_id))

        base_loss_cur_avg = engine.base_train_an_epoch(epoch_id, train_set, train_config)
        print('Epoch {} Done! time elapsed: {}, base_loss_cur_avg {:.4f}'.format(
            epoch_id,
            time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time)),
            base_loss_cur_avg))

        test_auc, test_logloss = engine.test(test_set, train_config)
        print('test_auc {:.4f}, test_logloss {:.4f}'.format(
            test_auc,
            test_logloss))
        print('time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))

        print('')

        # save checkpoint
        checkpoint_alias = 'Epoch{}_TestAUC{:.4f}_TestLOGLOSS{:.4f}.ckpt'.format(
            epoch_id,
            test_auc,
            test_logloss)
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_alias)
        saver.save(sess, checkpoint_path)


# build base model computation graph
base_model = MF(MF_hyperparams, train_config=train_config)

# create session
sess = tf.Session()

# create saver
saver = tf.train.Saver(max_to_keep=80)

orig_dir_name = train_config['dir_name']

for base_lr in [1e-2]:

    print('')
    print('base_lr', base_lr)

    train_config['base_lr'] = base_lr

    train_config['dir_name'] = orig_dir_name + '_' + str(base_lr)
    print('dir_name: ', train_config['dir_name'])

    # create current and next set
    train_set = data_df[(data_df['period'] >= train_config['train_start_period']) &
                        (data_df['period'] <= train_config['train_end_period'])]
    test_set = data_df[data_df['period'] == train_config['test_period']]
    train_config['train_set_size'] = len(train_set)
    train_config['test_set_size'] = len(test_set)
    print('train set size', len(train_set), 'test set size', len(test_set))

    # checkpoints directory
    checkpoints_dir = os.path.join('ckpts', train_config['dir_name'])
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # write train_config to text file
    with open(os.path.join(checkpoints_dir, 'config.txt'), mode='w') as f:
        f.write('train_config: ' + str(train_config) + '\n')
        f.write('\n')
        f.write('MF_hyperparams: ' + str(MF_hyperparams) + '\n')

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    train_base()
