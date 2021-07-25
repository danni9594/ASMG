import numpy as np
import pandas as pd
import glob


def process_cate(cate_ls):
    cate_lens = [len(cate) for cate in cate_ls]
    cate_seqs_matrix = np.zeros([len(cate_ls), max(cate_lens)], np.int32)
    i = 0
    for cateSeq in cate_ls:
        for j in range(len(cateSeq)):
            cate_seqs_matrix[i][j] = cateSeq[j]  # convert list of cateSeq into a matrix with zero padding
        i += 1
    return cate_seqs_matrix, cate_lens


class BatchLoader:
    """
    batch data loader by batch size
    return: [[users], [items], np.array(item_seqs_matrix), [seq_lens], [labels]] in batch iterator
    """

    def __init__(self, data_df, batch_size):

        self.data_df = data_df.reset_index(drop=True)  # df ['userId', 'itemId', 'label']
        self.data_df['index'] = self.data_df.index
        self.data_df['batch'] = self.data_df['index'].apply(lambda x: int(x / batch_size) + 1)
        self.num_batches = self.data_df['batch'].max()

    def get_batch(self, batch_id):

        batch = self.data_df[self.data_df['batch'] == batch_id]
        users = batch['userId'].tolist()
        items = batch['itemId'].tolist()
        labels = batch['label'].tolist()
        seq_lens = batch['itemSeq'].apply(len).tolist()

        item_seqs_matrix = np.zeros([len(batch), 30], np.int32)

        i = 0
        for itemSeq in batch['itemSeq'].tolist():
            for j in range(len(itemSeq)):
                item_seqs_matrix[i][j] = itemSeq[j]  # convert list of itemSeq into a matrix with zero padding
            i += 1

        return [users, items, item_seqs_matrix, seq_lens, labels]


def cal_roc_auc(scores, labels):

    arr = sorted(zip(scores, labels), key=lambda d: d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    if pos == 0 or neg == 0:
        return None

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp/neg, tp/pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        auc += ((x - prev_x) * (y + prev_y) / 2.)
        prev_x = x
        prev_y = y
    return auc


def cal_roc_gauc(users, scores, labels):
    # weighted sum of individual auc
    df = pd.DataFrame({'user': users,
                       'score': scores,
                       'label': labels})

    df_gb = df.groupby('user').agg(lambda x: x.tolist())

    auc_ls = []  # collect auc for all users
    user_imp_ls = []

    for row in df_gb.itertuples():
        auc = cal_roc_auc(row.score, row.label)
        if auc is None:
            pass
        else:
            auc_ls.append(auc)
            user_imp = len(row.label)
            user_imp_ls.append(user_imp)

    total_imp = sum(user_imp_ls)
    weighted_auc_ls = [auc * user_imp / total_imp for auc, user_imp in zip(auc_ls, user_imp_ls)]

    return sum(weighted_auc_ls)


def search_ckpt(search_alias, mode='last'):
    ckpt_ls = glob.glob(search_alias)

    if mode == 'best logloss':
        metrics_ls = [float(ckpt.split('.ckpt')[0].split('_')[-1].split('TestLOGLOSS')[-1]) for ckpt in ckpt_ls]  # logloss
        selected_metrics_pos_ls = [i for i, x in enumerate(metrics_ls) if x == min(metrics_ls)]  # find all positions of the selected ckpts
    elif mode == 'best auc':
        metrics_ls = [float(ckpt.split('.ckpt')[0].split('_')[-2].split('TestAUC')[-1]) for ckpt in ckpt_ls]  # auc
        selected_metrics_pos_ls = [i for i, x in enumerate(metrics_ls) if x == max(metrics_ls)]  # find all positions of the selected ckpts
    else:  # mode == 'last'
        metrics_ls = [float(ckpt.split('.ckpt')[0].split('_')[-3].split('Epoch')[-1]) for ckpt in ckpt_ls]  # epoch no.
        selected_metrics_pos_ls = [i for i, x in enumerate(metrics_ls) if x == max(metrics_ls)]  # find all positions of the selected ckpts
    ckpt = ckpt_ls[max(selected_metrics_pos_ls)]  # get the full path of the last selected ckpt

    ckpt = ckpt.split('.ckpt')[0]  # get the path name before .ckpt
    ckpt = ckpt + '.ckpt'  # get the path with .ckpt
    return ckpt
