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

train_config = {'method': 'pretrain',
                'dir_name': 'pretrain_train1-10_test11_10epoch',  # edit train test period range, number of epochs
                'train_start_date': 20141001,
                'train_end_date': 20141010,
                'test_date': 20141011,
                'train_set_size': None,  # train dataset size
                'test_set_size': None,  # test dataset size

                'base_optimizer': 'adam',  # base model optimizer: adam, rmsprop, sgd
                'base_lr': None,  # base model learning rate
                'base_bs': 1024,  # base model batch size
                'base_num_epochs': 10,  # base model number of epochs
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
base_model = EmbMLP(cates, cate_lens, EmbMLP_hyperparams, train_config=train_config)

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
    train_set = data_df[(data_df['date'] >= train_config['train_start_date']) &
                        (data_df['date'] <= train_config['train_end_date'])]
    test_set = data_df[data_df['date'] == train_config['test_date']]
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
        f.write('EmbMLP_hyperparams: ' + str(EmbMLP_hyperparams) + '\n')

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    train_base()
