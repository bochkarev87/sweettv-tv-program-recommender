"""Retrain on ALL logs (< END) with the configuration selected on validation (data/final_cfg.pkl)
and write the Kaggle submission for weeks 31-42 (candidate pool = shows in the 31-42 schedule)."""
import pickle
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from common import *
from models import *

inter, items, air, sub = load()
cfg = pickle.load(open(f'{DATA}/final_cfg.pkl', 'rb'))
item_ids = items['tv_show_id'].astype(int).values
item_index = {int(s): j for j, s in enumerate(item_ids)}
n_items = len(item_ids)

train = inter[inter['ts'] < END]
users = sub['user_id'].tolist()
pool = candidate_pool(air, TEST_START, pd.Timestamp('2100-01-01'))
pool_mask = np.zeros(n_items, bool)
pool_mask[[item_index[int(s)] for s in pool if int(s) in item_index]] = True

ui = user_item(train)
user_index = {u: i for i, u in enumerate(sorted(ui['user_id'].unique()))}
R = build_matrix(ui, user_index, item_index)
rows = np.array([user_index.get(u, len(user_index)) for u in users])
cold = np.array([u not in user_index for u in users])
take = lambda M: sp.vstack([M, sp.csr_matrix((1, n_items))]).tocsr()[rows]
pop_all = popularity_scores(ui, item_index)


def fill_cold(S):
    S = S.copy()
    S[cold] = pop_all[None, :]
    return S


def z(S):
    P = np.where(pool_mask[None, :], S, np.nan)
    return (S - np.nanmean(P, 1, keepdims=True)) / (np.nanstd(P, 1, keepdims=True) + 1e-9)


S_hist = fill_cold(take(history_matrix(train, END, user_index, item_index, cfg['hist_half_life'])).toarray())
U, s, Vt = svds(R.astype(float), k=cfg['svd_rank'])
S_mf = fill_cold((np.vstack([U, np.zeros((1, cfg['svd_rank']))])[rows] * s) @ Vt)
V = item_features(items, item_index, cfg['cbf_weights'], cfg['cbf_id_weight'])
S_cbf = fill_cold(cbf_scores(take(R), V))

print('final config:', {k: cfg[k] for k in ('hist_half_life', 'svd_rank', 'cbf_name', 'w_hist', 'w_mf', 'w_cbf')})
S = cfg['w_hist'] * z(S_hist) + cfg['w_mf'] * z(S_mf) + cfg['w_cbf'] * z(S_cbf)
OUT_FILE = 'submission.csv'
if 'air' in sys.argv:   # v2: schedule-based prior (how often a show is on air in the target window)
    air_cfg = pickle.load(open(f'{DATA}/final_cfg_air.pkl', 'rb'))
    win = air[air['ps'] >= TEST_START].copy()
    win['hours'] = (win['pe'] - win['ps']).dt.total_seconds() / 3600
    n_air, hours = np.zeros(n_items), np.zeros(n_items)
    for sid, g in win.groupby('tv_show_id'):
        if int(sid) in item_index:
            n_air[item_index[int(sid)]], hours[item_index[int(sid)]] = len(g), g['hours'].sum()
    prior = np.log1p(n_air if air_cfg['prior'] == 'log_n_airings' else hours)
    S = (air_cfg['w_hist'] * z(S_hist) + air_cfg['w_mf'] * z(S_mf)
         + air_cfg['w_prior'] * z(np.tile(prior, (len(users), 1))))
    OUT_FILE = 'submission_v2_airprior.csv'
    print('v2 config:', {k: air_cfg[k] for k in ('prior', 'w_hist', 'w_mf', 'w_prior')})
S = np.where(pool_mask[None, :], S + 1e-6 * pop_all / pop_all.max(), -np.inf)
top = np.argsort(-S, axis=1)[:, :K]

out = pd.DataFrame({'user_id': users, 'tv_show_id': [' '.join(str(int(item_ids[j])) for j in row) for row in top]})
out.to_csv(OUT_FILE, index=False)
print('pool size:', int(pool_mask.sum()), '| users:', len(out), '| cold users:', int(cold.sum()))
print(out.head())
tmpl = pd.read_csv(f'{D}\\submission.csv', dtype={'user_id': 'string'})
assert out['user_id'].tolist() == tmpl['user_id'].tolist(), 'user order differs from the template'
assert all(len(set(x.split())) == 5 for x in out['tv_show_id']), 'need 5 distinct shows per user'
print(OUT_FILE, 'OK:', out.shape)
