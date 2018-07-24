from feature import *
import pandas as pd


# userxitemでの総注文回数
def f000_n_item_orders(base: pd.DataFrame, prior: pd.DataFrame, path_base='../input/') -> pd.DataFrame:
    return make_agg_feature(base, prior, ['user_id', 'product_id'], 'order_number', 'count', 'n_item_orders')


# 最後の購買で買ったかどうか
def f001_last_ordered(base: pd.DataFrame, prior: pd.DataFrame, path_base='../input/') -> pd.DataFrame:
    last = last_purchase_ids(prior)

    last = pd.merge(last, prior, on=['user_id', 'order_number'], how='left')
    last['last_ordered'] = 1

    base_ = pd.merge(base, last[['last_ordered', 'user_id', 'product_id']], on=['user_id', 'product_id'],
                     how='left')
    base_['last_ordered'].fillna(0, inplace=True)
    return base_[['user_id', 'product_id', 'last_ordered']]


# そのアイテムの最終購買日からの経過日数
def f002_item_recency(base: pd.DataFrame, prior: pd.DataFrame, path_base='../input/') -> pd.DataFrame:
    last_order_numbers = prior.groupby(['user_id', 'product_id'])['order_number'].max().reset_index()

    orders = load_orders(path_base)
    orders['days_cum'] = orders.groupby(['user_id'])['days_since_prior_order'].cumsum()
    orders['days_cum'].fillna(0, inplace=True)

    o_ = orders.groupby('user_id')['days_since_prior_order'].sum().reset_index().rename(
        columns={'days_since_prior_order': 'total_days'})
    orders = pd.merge(orders, o_, on='user_id', how='left')
    orders['item_recency'] = orders['total_days'] - orders['days_cum']

    orders.head(100)

    last_order_numbers = pd.merge(last_order_numbers, orders[['item_recency', 'user_id', 'order_number']],
                                  on=['user_id', 'order_number'], how='left')

    return last_order_numbers[['item_recency', 'user_id', 'product_id']]


# 平均購買間隔
def f003_item_interval(base: pd.DataFrame, prior: pd.DataFrame, path_base='../input/') -> pd.DataFrame:
    pass

# 経過日数 / 平均購買間隔
def f004_item_recency_by_interval(base: pd.DataFrame, prior: pd.DataFrame, path_base='../input/') -> pd.DataFrame:
    pass

if __name__ == "__main__":
    df = load_base()
    all = load_all()
    prior = all[all.eval_set == 'prior']
    print(prior.shape)
    #make_feature(f000_n_item_orders, '../features/000_n_item_orders.f', all=prior, df=df)
    #make_feature(f001_last_ordered, '../features/001_last_ordered.f', all=prior, df=df)
    make_feature(f002_item_recency, '../features/002_item_recency.f', all=prior, df=df)
    #make_feature(f003_item_interval, '../features/003_item_interval.f', all=prior, df=df)

