from feature import *
import pandas as pd


# userの総オーダー数
def f100_n_orders(base: pd.DataFrame, all: pd.DataFrame, path_base='../input/') -> pd.DataFrame:
    agg = make_agg_feature(base, all, ['user_id'], 'order_number', 'nunique', 'n_orders')
    print(agg.head())
    print(agg.info())
    return agg


# userの最終購買からの経過日数
def f101_recency(base: pd.DataFrame, all: pd.DataFrame, path_base='../input/') -> pd.DataFrame:
    orders = load_orders(path_base)
    last = orders[orders.eval_set != 'prior'].rename(columns={'days_since_prior_order': 'recency'})

    return last[['user_id', 'recency']]


def f102_reorder_rate(base: pd.DataFrame, prior: pd.DataFrame, path_base='../input/') -> pd.DataFrame:
    return make_agg_feature(base, all, ['user_id'], 'reordered', 'mean', 'reorder_rate')


if __name__ == "__main__":
    df = load_base()
    all = load_all()

    #make_feature(f100_n_orders, '../features/f100_n_orders.f', all=all, df=df)
    make_feature(f101_recency, '../features/f101_recency.f', all=all, df=df)
    #make_feature(f102_reorder_rate, '../features/f102_reorder_rate.f', all=all, df=df)
