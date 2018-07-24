import gc
import pandas as pd
import lightgbm as lgb
import numpy as np
import time
from contextlib import contextmanager

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


class Model(object):
    def __init__(self, user_features=None, user_item_features=None, save_head=True, debug=False):
        x = pd.read_feather('../input/base.f')

        if debug:
            x = x.head(1000)
        print(x.shape)

        if user_features is not None:
            for f in user_features:
                print('merge {}...'.format(f))
                df = pd.read_feather('../features/' + f)
                x = pd.merge(x, df, on='user_id', how='left')

        x.head(1000).to_csv('x.csv')

        if user_item_features is not None:
            for f in user_item_features:
                print('merge {}...'.format(f))
                df = pd.read_feather('../features/' + f)
                x = pd.merge(x, df, on=['user_id', 'product_id'], how='left')

        print(x.shape)

        with timer('split data'):
            x.set_index('user_id', inplace=True)
            x['product_id'] = x['product_id'].astype(np.int32)

            self.x_train = x[x['is_train']].drop('is_train', axis=1)
            self.x_test = x[~x['is_train']].drop('is_train', axis=1)
            del x
            gc.collect()

            print('train: {}, test: {}'.format(self.x_train.shape, self.x_test.shape))

        print(self.x_train.head())

        if save_head:
            self.x_train.head(100).to_csv('x_train.head100.csv')
            self.x_test.head(100).to_csv('x_test.head100.csv')

        with timer('make lgb.Dataset'):
            self.dtrain = lgb.Dataset(self.x_train.drop(['target'], axis=1), self.x_train['target'])

        self.params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'metric': 'binary_logloss',
            'num_leaves': 15,
            'seed': 0,
            'learning_rate': 0.1
        }

        orders = pd.read_feather('../input/orders.csv.f')

        self.test_users = orders[['eval_set', 'user_id', 'order_id']][orders['eval_set'] == 'test']

    def train(self, boost_round):
        self.booster = lgb.train(self.params, self.dtrain, num_boost_round=boost_round)

        predicted = self.x_test.reset_index()[['user_id', 'product_id']]
        predicted = pd.merge(predicted, self.test_users, on='user_id', how='left')

        predicted['y'] = self.booster.predict(self.x_test.drop('target', axis=1))

        print(predicted.head())

        th = 0.2  # TODO

        predicted['product_id'] = predicted['product_id'].astype(str)

        # 閾値を超えた行だけを抜き出して、submit用に整形
        sub = predicted[predicted['y'] > th].groupby(['order_id'])['product_id'] \
            .apply(lambda x: ' '.join(x)) \
            .reset_index() \
            .rename(columns={'product_id': 'products'})

        # 閾値をひとつも超えなかったオーダーが抜けてしまうので、Join->fillnaでNoneを埋める
        sub = pd.merge(self.test_users[['order_id']], sub, on='order_id', how='left')
        sub.fillna('None', inplace=True)

        sub.sort_values(by='order_id', inplace=True)
        sub.head()

        return sub


if __name__ == "__main__":
    m = Model(user_features=['f100_n_orders.f', 'f101_recency.f', 'f102_reorder_rate.f'],
              user_item_features=['000_n_item_orders.f', '001_last_ordered.f', '002_item_recency.f'], save_head=True, debug=False)

    sub = m.train(300)
    sub.to_csv('../output/baseline.csv',index=False)
