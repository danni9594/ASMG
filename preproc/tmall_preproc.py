import pandas as pd
import random
random.seed(1234)

# convert csv into pandas dataframe
df = pd.read_csv('../raw_data/data_format1/user_log_format1.csv')
df['date'] = df['time_stamp'].apply(lambda x: 20140000 + x)
df = df[df['action_type'] == 0]  # keep click record only
df = df.drop(['seller_id', 'brand_id', 'time_stamp', 'action_type'], axis=1)
df.columns = ['userId', 'itemId', 'cateId', 'date']  # rename

# extract 31 days data
start_date = 20141001
end_date = 20141031
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# keep top 50000 users with most interactions
user_counts = df['userId'].value_counts()
user_counts = user_counts.sort_values(ascending=False)
users_to_keep = user_counts[:50000].index
df = df[df.userId.isin(users_to_keep)]

# filter out items with less than 20 interactions
item_counts = df['itemId'].value_counts()
items_to_keep = item_counts[item_counts >= 20].index
df = df[df.itemId.isin(items_to_keep)]

# remap id
user_id = sorted(df['userId'].unique().tolist())  # sort column
user_map = dict(zip(user_id, range(len(user_id))))  # create map, key is original id, value is mapped id starting from 0
df['userId'] = df['userId'].map(lambda x: user_map[x])  # map key to value in df

item_id = sorted(df['itemId'].unique().tolist())  # sort column
item_map = dict(zip(item_id, range(len(item_id))))  # create map, key is original id, value is mapped id starting from 0
df['itemId'] = df['itemId'].map(lambda x: item_map[x])  # map key to value in df

cate_id = sorted(df['cateId'].unique().tolist())  # sort column
cate_map = dict(zip(cate_id, range(len(cate_id))))  # create map, key is original id, value is mapped id starting from 0
df['cateId'] = df['cateId'].map(lambda x: cate_map[x])  # map key to value in df

print(df.head(20))
print('num_users: ', len(user_map))  # 49986
print('num_items: ', len(item_map))  # 43571
print('num_cates: ', len(cate_map))  # 634
print('num_records: ', len(df))  # 3047101

# create item meta df
meta_df = df.drop(['userId', 'date'], axis=1).drop_duplicates()
meta_df = meta_df.drop_duplicates(subset='itemId')  # some items map to multiple cates
meta_df = meta_df.sort_values(['itemId'], ascending=True).reset_index(drop=True)
assert len(meta_df) == len(item_map)
print(meta_df.head())

# save csv
# ['itemId', 'cateId']
meta_df.to_csv('../datasets/tmall_1mth_2014_item20user50k_1neg_30seq_item_meta.csv', index=False)


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
    user_history_df = user_df[user_df['date'] <= row.date].sort_values(['date'], ascending=False).reset_index(drop=True)
    userHist = user_history_df['itemId'].unique().tolist()
    neg_lss.append(gen_neg(len(item_map), userHist, num_neg))

    user_history_df = user_history_df[user_history_df['date'] < row.date].sort_values(['date'], ascending=False).reset_index(drop=True)
    item_seq_ls = user_history_df['itemId'][:max_len].tolist()
    itemSeq = '#'.join(str(i) for i in item_seq_ls)
    item_seqs.append(itemSeq)

    count += 1
    if count % 500000 == 0:
        print('done row {}'.format(count))

df['neg_itemId_ls'] = neg_lss
df['itemSeq'] = item_seqs

users, itemseqs, items, labels, dates = [], [], [], [], []
for row in df.itertuples():
    users.append(row.userId)
    itemseqs.append(row.itemSeq)
    items.append(row.itemId)
    labels.append(1)  # positive samples have label 1
    dates.append(row.date)
    for j in range(num_neg):
        users.append(row.userId)
        itemseqs.append(row.itemSeq)
        items.append(row.neg_itemId_ls[j])
        labels.append(0)  # negative samples have label 0
        dates.append(row.date)

df = pd.DataFrame({'userId': users,
                   'itemSeq': itemseqs,
                   'itemId': items,
                   'label': labels,
                   'date': dates})

print(df.head(20))
print(len(df))  # 6094202

# save csv
# ['userId', 'itemSeq', 'itemId', 'label', 'date']
df.to_csv('../datasets/tmall_1mth_2014_item20user50k_1neg_30seq.csv', index=False)
