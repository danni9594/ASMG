import pandas as pd
import random
import datetime
random.seed(1234)

# convert csv into pandas dataframe
df = pd.read_csv('../raw_data/ml-25m/ratings.csv')
df['date'] = df['timestamp'].apply(lambda x: int(datetime.datetime.fromtimestamp(x).strftime('%Y%m%d')))

# preprocess
df['label'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
df = df.drop(['rating'], axis=1)  # drop useless
df.columns = ['userId', 'itemId', 'timestamp', 'date', 'label']  # rename

# extract 5 years data
start_date = 20140101
end_date = 20181231
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# remap id
user_id = sorted(df['userId'].unique().tolist())  # sort column
user_map = dict(zip(user_id, range(len(user_id))))  # create map, key is original id, value is mapped id starting from 0
df['userId'] = df['userId'].map(lambda x: user_map[x])  # map key to value in df

item_id = sorted(df['itemId'].unique().tolist())  # sort column
item_map = dict(zip(item_id, range(len(item_id))))  # create map, key is original id, value is mapped id starting from 0
df['itemId'] = df['itemId'].map(lambda x: item_map[x])  # map key to value in df

print(df.head(20))
print('num_users: ', len(user_map))  # 43181
print('num_items: ', len(item_map))  # 51142
print('num_records: ', len(df))  # 6840091

# collect user history
df_gb = df.groupby(['userId'])
item_seqs = []
max_len = 30
count = 0
for row in df.itertuples():
    user_df = df_gb.get_group(row.userId)
    user_df = user_df[user_df['label'] == 1]
    user_history_df = user_df[user_df['timestamp'] < row.timestamp].sort_values(['timestamp'], ascending=False).reset_index(drop=True)
    item_seq_ls = user_history_df['itemId'][:max_len].tolist()
    itemSeq = '#'.join(str(i) for i in item_seq_ls)
    item_seqs.append(itemSeq)
    count += 1
    if count % 500000 == 0:
        print('done row {}'.format(count))
df['itemSeq'] = item_seqs

df = df[['userId', 'itemSeq', 'itemId', 'label', 'date', 'timestamp']]  # switch columns

print(df.head(20))
print(len(df))  # 6840091

# save csv
# ['userId', 'itemSeq', 'itemId', 'label', 'date', 'timestamp']
df.to_csv('../datasets/ml_5yr_2014_2018_30seq.csv', index=False)

# create movie meta df
meta_df = pd.read_csv('/Users/ali-226125n/PycharmProjects/raw_datasets/ml-25m/movies.csv')
genres = meta_df['genres'].tolist()  # 62423
genre_ls = []
for genre in genres:
    genre_ls.extend(genre.split('|'))
genre_ls = list(set(genre_ls))  # ['Sci-Fi', 'Comedy', 'IMAX', '(no genres listed)', 'Romance', 'Mystery', 'Film-Noir', 'Action', 'Adventure', 'Thriller', 'Animation', 'Crime', 'Children', 'War', 'Drama', 'Documentary', 'Horror', 'Western', 'Fantasy', 'Musical']

genre_ls = sorted(genre_ls)  # sort ls
cate_map = dict(zip(genre_ls, range(len(genre_ls))))  # create map, key is original id, value is mapped id starting from 0
meta_df = meta_df[meta_df['movieId'].isin(item_map.keys())]  # some movies are not included
meta_df['cateId'] = meta_df['genres'].map(lambda x: '#'.join(str(cate_map[i]) for i in x.split('|')))  # map key to value in mata_df
meta_df['itemId'] = meta_df['movieId'].map(lambda x: item_map[x])  # map key to value in mata_df
meta_df = meta_df.drop(['movieId', 'title', 'genres'], axis=1)  # drop useless
meta_df = meta_df.sort_values(['itemId'], ascending=True).reset_index(drop=True)
meta_df = meta_df[['itemId', 'cateId']]  # switch columns

print(meta_df.head())

# save csv
# ['itemId', 'cateId']
meta_df.to_csv('../datasets/ml_5yr_2014_2018_30seq_item_meta.csv', index=False)

