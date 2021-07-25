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

train_config = {'method': 'SPMF_by_period',
                'dir_name': 'SPMF_2_train11-23_test24-30_1epoch',  # edit strategy type, train test period, number of epochs
                'pretrain_model': 'pretrain_train1-10_test11_10epoch_0.001',
                'start_date': 20140101,  # overall train start date
                'end_date': 20181231,  # overall train end date
                'num_periods': 31,  # number of periods divided into
                'train_start_period': 11,
                'test_start_period': 24,
                'cur_period': None,  # current incremental period
                'next_period': None,  # next incremental period
                'cur_set_size': None,  # current incremental dataset size
                'next_set_size': None,  # next incremental dataset size
                'period_alias': None,  # individual period directory alias to save ckpts
                'restored_ckpt_mode': 'best auc',  # mode to search the checkpoint to restore, 'best auc', 'best gauc', 'last'
                'restored_ckpt': None,  # configure in the for loop

                'strategy': 2,  # two different sampling strategies
                'frac_of_pretrain_D': None,  # reservoir size as a fraction of pretrain dataset, less than or equal to 1
                'res_cur_ratio': None,  # the ratio of reservoir sample to current set, only for strategy 2

                'base_optimizer': 'adam',  # base model optimizer: adam, rmsprop, sgd
                'base_lr': None,  # base model learning rate
                'base_bs': 1024,  # base model batch size
                'base_num_epochs': 1,  # base model number of epochs
                'shuffle': False,  # whether to shuffle the dataset for each epoch
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


def compute_prob_and_gen_set_and_update_reservoir():

    """
    this strategy follows exactly the method from the paper "Streaming ranking based recommender systems"
    train_set = samples of (current_set + reservoir)
    """
    compute_prob_start_time = time.time()

    pos_train_set = pd.concat([reservoir, pos_cur_set], ignore_index=False)  # combine R and W
    neg_train_set = pd.concat([neg_reservoir, neg_cur_set], ignore_index=False)  # combine R and W

    # compute prob
    pos_train_batch_loader = BatchLoader(pos_train_set, train_config['base_bs'])

    scores = []
    for i in range(1, pos_train_batch_loader.num_batches + 1):
        pos_train_batch = pos_train_batch_loader.get_batch(batch_id=i)
        batch_scores, batch_losses = base_model.inference(sess, pos_train_batch)  # sess.run
        scores.extend(batch_scores)

    ordered_pos_train_set = pos_train_set
    ordered_pos_train_set['score'] = scores
    ordered_pos_train_set = ordered_pos_train_set.sort_values(['score'], ascending=False).reset_index(drop=True)  # edit
    ordered_pos_train_set['rank'] = np.arange(len(ordered_pos_train_set))
    total_num = len(pos_train_set)
    ordered_pos_train_set['weight'] = ordered_pos_train_set['rank'].apply(lambda x: np.exp(x / total_num))
    total_weights = ordered_pos_train_set['weight'].sum()
    ordered_pos_train_set['prob'] = ordered_pos_train_set['weight'].apply(lambda x: x / total_weights)
    ordered_pos_train_set = ordered_pos_train_set.drop(['score', 'rank', 'weight'], axis=1)

    # generate train set
    sampled_pos_train_set = ordered_pos_train_set.sample(n=len(pos_cur_set), replace=False, weights='prob')
    sampled_pos_train_set = sampled_pos_train_set.drop(['prob'], axis=1)
    sampled_neg_train_set = neg_train_set.sample(n=len(neg_cur_set), replace=False)
    sampled_train_set = pd.concat([sampled_pos_train_set, sampled_neg_train_set], ignore_index=False)
    sampled_train_set = sampled_train_set.sort_values(['period']).reset_index(drop=True)

    # update pos reservoir
    t = len(data_df[(data_df['period'] < train_config['cur_period']) & (data_df['label'] == 1)])
    probs_to_res = len(reservoir) / (t + np.arange(len(pos_cur_set)) + 1)
    random_probs = np.random.rand(len(pos_cur_set))
    selected_pos_cur_set = pos_cur_set[probs_to_res > random_probs]
    num_left_in_res = len(reservoir) - len(selected_pos_cur_set)
    updated_reservoir = pd.concat([reservoir.sample(n=num_left_in_res), selected_pos_cur_set], ignore_index=False)
    print('selected_pos_cur_set size', len(selected_pos_cur_set))
    # print('num_in_res', len(reservoir))
    # print('num_left_in_res', num_left_in_res)
    # print('num_in_updated_res', len(updated_reservoir))

    # update neg reservoir
    t = len(data_df[(data_df['period'] < train_config['cur_period']) & (data_df['label'] == 0)])
    probs_to_res = len(neg_reservoir) / (t + np.arange(len(neg_cur_set)) + 1)
    random_probs = np.random.rand(len(neg_cur_set))
    selected_neg_cur_set = neg_cur_set[probs_to_res > random_probs]
    num_left_in_res = len(neg_reservoir) - len(selected_neg_cur_set)
    updated_neg_reservoir = pd.concat([neg_reservoir.sample(n=num_left_in_res), selected_neg_cur_set], ignore_index=False)
    print('selected_neg_cur_set size', len(selected_neg_cur_set))
    # print('num_in_neg_res', len(neg_reservoir))
    # print('num_left_in_neg_res', num_left_in_res)
    # print('num_in_updated_neg_res', len(updated_neg_reservoir))

    print('compute prob and generate train set and update reservoir time elapsed: {}'.format(
        time.strftime('%H:%M:%S', time.gmtime(time.time() - compute_prob_start_time))))

    return sampled_train_set, updated_reservoir, updated_neg_reservoir


def compute_prob_and_gen_set_and_update_reservoir2():
    """
    this strategy modify slightly the method from paper "Streaming ranking based recommender systems"
    train_set = current_set + samples of reservoir (need to set ratio of reservoir sample to current set)
    """
    compute_prob_start_time = time.time()

    # compute prob
    reservoir_batch_loader = BatchLoader(reservoir, train_config['base_bs'])

    scores = []
    for i in range(1, reservoir_batch_loader.num_batches + 1):
        reservoir_batch = reservoir_batch_loader.get_batch(batch_id=i)
        batch_scores, batch_losses = base_model.inference(sess, reservoir_batch)  # sess.run
        scores.extend(batch_scores.tolist())

    ordered_reservoir = reservoir
    ordered_reservoir['score'] = scores
    ordered_reservoir = ordered_reservoir.sort_values(['score'], ascending=False).reset_index(drop=True)  # edit
    ordered_reservoir['rank'] = np.arange(len(ordered_reservoir))
    total_num = len(reservoir)
    ordered_reservoir['weight'] = ordered_reservoir['rank'].apply(lambda x: np.exp(x / total_num))
    total_weights = ordered_reservoir['weight'].sum()
    ordered_reservoir['prob'] = ordered_reservoir['weight'].apply(lambda x: x / total_weights)
    ordered_reservoir = ordered_reservoir.drop(['score', 'rank', 'weight'], axis=1)

    # generate train set
    sampled_pos_reservoir = ordered_reservoir.sample(n=int(len(pos_cur_set) * train_config['res_cur_ratio']), replace=False, weights='prob')
    sampled_pos_reservoir = sampled_pos_reservoir.drop(['prob'], axis=1)
    sampled_neg_reservoir = neg_reservoir.sample(n=int(len(neg_cur_set) * train_config['res_cur_ratio']), replace=False)
    sampled_reservoir = pd.concat([sampled_pos_reservoir, sampled_neg_reservoir], ignore_index=False)
    sampled_train_set = pd.concat([sampled_reservoir, cur_set], ignore_index=False)
    sampled_train_set = sampled_train_set.sort_values(['period']).reset_index(drop=True)
    print('sampled_reservoir size', len(sampled_reservoir))
    # print('sampled_train_set size', len(sampled_train_set))

    # update reservoir
    t = len(data_df[(data_df['period'] < train_config['cur_period']) & (data_df['label'] == 1)])
    probs_to_res = len(reservoir) / (t + np.arange(len(pos_cur_set)) + 1)
    random_probs = np.random.rand(len(pos_cur_set))
    selected_pos_cur_set = pos_cur_set[probs_to_res > random_probs]
    num_left_in_res = len(reservoir) - len(selected_pos_cur_set)
    updated_reservoir = pd.concat([reservoir.sample(n=num_left_in_res), selected_pos_cur_set], ignore_index=False)
    print('selected_pos_current_set size', len(selected_pos_cur_set))
    # print('num_in_res', len(reservoir))
    # print('num_left_in_res', num_left_in_res)
    # print('num_in_updated_res', len(updated_reservoir))

    # update neg reservoir
    t = len(data_df[(data_df['period'] < train_config['cur_period']) & (data_df['label'] == 0)])
    probs_to_res = len(neg_reservoir) / (t + np.arange(len(neg_cur_set)) + 1)
    random_probs = np.random.rand(len(neg_cur_set))
    selected_neg_cur_set = neg_cur_set[probs_to_res > random_probs]
    num_left_in_res = len(neg_reservoir) - len(selected_neg_cur_set)
    updated_neg_reservoir = pd.concat([neg_reservoir.sample(n=num_left_in_res), selected_neg_cur_set], ignore_index=False)
    print('selected_neg_cur_set size', len(selected_neg_cur_set))
    # print('num_in_neg_res', len(neg_reservoir))
    # print('num_left_in_neg_res', num_left_in_res)
    # print('num_in_updated_neg_res', len(updated_neg_reservoir))

    print('compute prob and generate train set and update reservoir time elapsed: {}'.format(
        time.strftime('%H:%M:%S', time.gmtime(time.time() - compute_prob_start_time))))

    return sampled_train_set, updated_reservoir, updated_neg_reservoir


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

for frac in [0.3]:

    for ratio in [0.5]:

        for base_lr in [1e-3]:

            print('')
            print('frac_of_pretrain_D', frac, 'res_cur_ratio', ratio, 'base_lr', base_lr)

            train_config['frac_of_pretrain_D'] = frac
            train_config['res_cur_ratio'] = ratio
            train_config['base_lr'] = base_lr

            train_config['dir_name'] = orig_dir_name + '_' + str(frac) + '_' + str(ratio) + '_' + str(base_lr)  # for strategy 2
            # train_config['dir_name'] = orig_dir_name + '_' + str(frac) + '_' + str(base_lr)  # for strategy 1
            print('dir_name: ', train_config['dir_name'])

            test_aucs = []
            test_loglosses = []

            for i in range(train_config['train_start_period'], train_config['num_periods']):

                # configure cur_period, next_period
                train_config['cur_period'] = i
                train_config['next_period'] = i + 1
                print('')
                print('current period: {}, next period: {}'.format(
                    train_config['cur_period'],
                    train_config['next_period']))
                print('')

                # create current and next set
                cur_set = data_df[data_df['period'] == train_config['cur_period']]
                next_set = data_df[data_df['period'] == train_config['next_period']]
                train_config['cur_set_size'] = len(cur_set)
                train_config['next_set_size'] = len(next_set)
                print('current set size', len(cur_set), 'next set size', len(next_set))

                # create train
                pos_cur_set = cur_set[cur_set['label'] == 1]
                neg_cur_set = cur_set[cur_set['label'] == 0]

                if i == train_config['train_start_period']:
                    pos_pretrain_set = data_df[(data_df['period'] < train_config['train_start_period']) & (data_df['label'] == 1)]
                    reservoir_size = int(len(pos_pretrain_set) * train_config['frac_of_pretrain_D'])
                    reservoir = pos_pretrain_set.sample(n=reservoir_size)

                    neg_pretrain_set = data_df[(data_df['period'] < train_config['train_start_period']) & (data_df['label'] == 0)]
                    neg_reservoir_size = int(len(neg_pretrain_set) * train_config['frac_of_pretrain_D'])
                    neg_reservoir = neg_pretrain_set.sample(n=neg_reservoir_size)

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

                    if train_config['strategy'] == 1:
                        cur_set, reservoir, neg_reservoir = compute_prob_and_gen_set_and_update_reservoir()
                    else:  # train_config['strategy'] == 2
                        cur_set, reservoir, neg_reservoir = compute_prob_and_gen_set_and_update_reservoir2()

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
