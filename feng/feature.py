import pandas as pd
import numpy as np
import os
import time
from contextlib import contextmanager
from typing import List, Union


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# まずfeatherに変換しておく
def load(path):
    if not os.path.exists(path + '.f'):
        pd.read_csv(path).to_feather(path + '.f')
    return pd.read_feather(path + '.f')


def make_agg_feature(base: pd.DataFrame, all: pd.DataFrame, group: Union[List[str], str], target: str, aggregate: str, feature_name: str):
    all_agg = all.groupby(group)[target].agg(aggregate).reset_index().rename(columns={target: feature_name})
    return all_agg


def prepare_dataset(path_base='../input/'):
    with timer('load data'):
        prior = load(path_base + 'order_products__prior.csv')
        train = load(path_base + 'order_products__train.csv')
        orders = load(path_base + 'orders.csv')

    # order-id/user-id/product-idを一つにまとめる
    with timer('merge & drop duplicates'):
        prior_orders = pd.merge(prior, orders[['order_id', 'user_id']], on='order_id', how='left')
        print(prior_orders.shape)
        prior_orders.drop_duplicates(subset=['user_id', 'product_id'], inplace=True)
        print(prior_orders.shape)

    # userごとに、過去買ったアイテムをまとめる
    with timer('aggregate prior products'):
        prior_orders['product_id_str'] = prior_orders['product_id'].astype(str)
        prior_products = prior_orders.groupby('user_id')['product_id_str'].apply(lambda x: ' '.join(x)).reset_index()

    with timer('extract last order for each user-x-item'):
        # 全データをまとめる
        all = pd.merge(orders, pd.concat([train,prior]), on='order_id', how='left')
        all.head()

        # user x itemで最後のデータだけを残す
        last_order_by_user_x_item = all.drop_duplicates(subset=['user_id','product_id'], keep='last')
        last_order_by_user_x_item.head()

    # eval_set == trainかつreorder == 1がターゲット
    with timer('make user_id x product_id x target x is_train'):
        # train/testでユーザーが振り分けられているので、trainに属するuserかどうかを列に追加
        train_users = orders[['eval_set', 'user_id']][orders['eval_set'] == 'train']

        X = last_order_by_user_x_item[['user_id', 'product_id', 'eval_set', 'reordered']].dropna()
        X['is_train'] = X['user_id'].isin(train_users['user_id'])
        X['target'] = ((X['eval_set'] == 'train') & (X['reordered'] == 1)).astype(np.int32)
        X.drop(['eval_set', 'reordered'], axis=1, inplace=True)

        print(X['target'].value_counts())
        print(X.head())

    return X


def make_all_df(path_base='../input/'):
    with timer('load data'):
        aisles = load(path_base+'aisles.csv')
        departments = load(path_base+'departments.csv')
        prior = load(path_base+'order_products__prior.csv')
        train = load(path_base+'order_products__train.csv')
        orders = load(path_base+'orders.csv')
        products = load(path_base+'products.csv')

    with timer('extract last order for each user-x-item'):
        # 全データをまとめる
        all = pd.merge(orders, pd.concat([train, prior]), on='order_id', how='left')
        products = pd.merge(products, aisles, on='aisle_id', how='left')
        products = pd.merge(products, departments, on='department_id', how='left')
        all = pd.merge(all, products, on='product_id', how='left')
        return all


def load_base(path_base='../input/'):
    return pd.read_feather(path_base+'base.f')


def load_all(path_base='../input/'):
    return pd.read_feather(path_base+'all.f')


def make_feature(func, path, all, df, path_base='../input/'):
    x = func(df, all)
    x.reset_index(drop=True).to_feather(path)
    x.head(100).to_csv(path+'.sample.csv', index=False)


def last_purchase_ids(prior: pd.DataFrame) -> pd.DataFrame:
    return prior.groupby('user_id')['order_number'].max().reset_index()


def load_orders(path_base='../input/'):
    return load(path_base+'orders.csv')

if __name__ == "__main__":
    #x = prepare_dataset().reset_index()
    #x.head(1000).to_csv('../input/base_sample.csv', index=False)
    #x.to_feather('../input/base.f')

    all = make_all_df().reset_index()
    all.head(1000).to_csv('../input/all_sample.csv', index=False)
    all.to_feather('../input/all.f')
