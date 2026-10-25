"""Shared protocol: temporal split, candidate pool, ground truth, MAP@5."""
import os
import numpy as np
import pandas as pd

D = r'C:\Users\swaggaf0x\.cache\kagglehub\competitions\sweettv-tv-program-recommender'
DATA = 'data'
CUTOFF = pd.Timestamp(os.environ.get('SWEET_CUTOFF', '2020-06-01'))    # train: ts < CUTOFF ; validation: CUTOFF <= ts < END
END = pd.Timestamp('2020-07-27')       # end of the available logs
TEST_START = pd.Timestamp('2020-07-27')  # schedule of weeks 31-42 starts here
K = 5


def load():
    inter = pd.read_pickle(f'{DATA}/interactions.pkl')
    items = pd.read_pickle(f'{DATA}/items.pkl')
    air = pd.read_pickle(f'{DATA}/airings.pkl')
    sub = pd.read_csv(f'{D}\\submission.csv', dtype={'user_id': 'string'})
    return inter, items, air, sub


def user_item(inter):
    """Aggregate sessions to (user, show): total watch seconds, #sessions, last time."""
    return (inter.groupby(['user_id', 'tv_show_id'])
            .agg(sec=('overlap_sec', 'sum'), n=('ts', 'size'), last=('ts', 'max'))
            .reset_index())


def candidate_pool(air, start, end):
    """Shows that are on air (non-zero id) in [start, end)."""
    a = air[(air['ps'] >= start) & (air['ps'] < end)]
    return np.sort(a['tv_show_id'].unique())


def make_truth(val_inter, users, k=K):
    """Per user: the k shows with largest total watch time in the validation window."""
    ui = user_item(val_inter[val_inter['user_id'].isin(users)])
    ui = ui.sort_values(['user_id', 'sec', 'tv_show_id'], ascending=[True, False, True])
    top = ui.groupby('user_id').head(k)
    return top.groupby('user_id')['tv_show_id'].apply(list).to_dict()


def apk(rec, truth, k=K):
    truth = set(truth)
    hits, s = 0, 0.0
    for i, r in enumerate(rec[:k], 1):
        if r in truth:
            hits += 1
            s += hits / i
    return s / min(len(truth), k)


def evaluate(recs, truths, k=K):
    users = list(truths)
    ap = np.array([apk(recs[u], truths[u], k) for u in users])
    hr = np.array([len(set(recs[u][:k]) & set(truths[u])) > 0 for u in users])
    prec = np.array([len(set(recs[u][:k]) & set(truths[u])) / k for u in users])
    return {'MAP@5': ap.mean(), 'HR@5': hr.mean(), 'P@5': prec.mean()}
