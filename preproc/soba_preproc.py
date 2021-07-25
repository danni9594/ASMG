import pandas as pd
import random
import datetime
random.seed(1234)

# convert csv into pandas dataframe
df = pd.read_csv('../raw_data/Sobazaar-hashID.csv.gz')

# preprocess
df['date'] = df['Timestamp'].apply(lambda x: int(''.join(c for c in x.split('T')[0] if c.isdigit())))  # extract date and convert to int
df['timestamp'] = df['Timestamp'].apply(lambda x: int(datetime.datetime.strptime(x.split('.')[0], '%Y-%m-%dT%H:%M:%S').timestamp()))
df = df.drop(['Action', 'Timestamp'], axis=1)  # drop useless
df.columns = ['itemId', 'userId', 'date', 'timestamp']  # rename
df = df[['userId', 'itemId', 'date', 'timestamp']]  # switch columns

# remap id
user_id = sorted(df['userId'].unique().tolist())  # sort column
user_map = dict(zip(user_id, range(len(user_id))))  # create map, key is original id, value is mapped id starting from 0
df['userId'] = df['userId'].map(lambda x: user_map[x])  # map key to value in df

item_id = sorted(df['itemId'].unique().tolist())  # sort column
item_map = dict(zip(item_id, range(len(item_id))))  # create map, key is original id, value is mapped id starting from 0
df['itemId'] = df['itemId'].map(lambda x: item_map[x])  # map key to value in df

print(df.head(20))
print('num_users: ', len(user_map))  # 17126
print('num_items: ', len(item_map))  # 24785
print('num_records: ', len(df))  # 842660


def gen_neg(num_items, pos_ls, num_neg):
    neg_ls = []
    for n in range(num_neg):  # generate num_neg
        neg = pos_ls[0]
        while neg in pos_ls:
            neg = random.randint(0, num_items - 1)
        neg_ls.append(neg)
    return neg_ls


# collect user history
df_gb = df.groupby(['userId'])
neg_lss = []
num_neg = 1
item_seqs = []
max_len = 30
count = 0
for row in df.itertuples():
    user_df = df_gb.get_group(row.userId)
    user_history_df = user_df[user_df['timestamp'] <= row.timestamp].sort_values(['timestamp'], ascending=False).reset_index(drop=True)
    userHist = user_history_df['itemId'].unique().tolist()
    neg_lss.append(gen_neg(len(item_map), userHist, num_neg))

    user_history_df = user_history_df[user_history_df['timestamp'] < row.timestamp].sort_values(['timestamp'], ascending=False).reset_index(drop=True)
    item_seq_ls = user_history_df['itemId'][:max_len].tolist()
    itemSeq = '#'.join(str(i) for i in item_seq_ls)
    item_seqs.append(itemSeq)

    count += 1
    if count % 100000 == 0:
        print('done row {}'.format(count))

df['neg_itemId_ls'] = neg_lss
df['itemSeq'] = item_seqs

users, itemseqs, items, labels, dates, timestamps = [], [], [], [], [], []
for row in df.itertuples():
    users.append(row.userId)
    itemseqs.append(row.itemSeq)
    items.append(row.itemId)
    labels.append(1)  # positive samples have label 1
    dates.append(row.date)
    timestamps.append(row.timestamp)
    for j in range(num_neg):
        users.append(row.userId)
        itemseqs.append(row.itemSeq)
        items.append(row.neg_itemId_ls[j])
        labels.append(0)  # negative samples have label 0
        dates.append(row.date)
        timestamps.append(row.timestamp)

df = pd.DataFrame({'userId': users,
                   'itemSeq': itemseqs,
                   'itemId': items,
                   'label': labels,
                   'date': dates,
                   'timestamp': timestamps})

print(df.head(20))
print(len(df))  # 1685320

# save csv and pickle
# ['userId', 'itemSeq', 'itemId', 'label', 'date', 'timestamp']
df.to_csv('../datasets/soba_4mth_2014_1neg_30seq_1.csv', index=False)
