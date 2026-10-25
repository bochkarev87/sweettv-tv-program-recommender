"""Validation experiments: popularity vs own-history vs CBF vs MF (ALS, SVD) + hybrid.
Same temporal split, candidate pool and protocol for every method (see common.py)."""
import time
import pickle
import itertools
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from common import *
from models import *

inter, items, air, sub = load()
item_ids = items['tv_show_id'].astype(int).values
item_index = {int(s): j for j, s in enumerate(item_ids)}
n_items = len(item_ids)

train = inter[inter['ts'] < CUTOFF]
val = inter[(inter['ts'] >= CUTOFF) & (inter['ts'] < END)]
sub_users = sub['user_id'].tolist()
truths = make_truth(val, sub_users)
eval_users = [u for u in sub_users if u in truths]

pool = candidate_pool(air, CUTOFF, END)
pool_mask = np.zeros(n_items, bool)
pool_mask[[item_index[int(s)] for s in pool]] = True

ui_train = user_item(train)
user_index = {u: i for i, u in enumerate(sorted(ui_train['user_id'].unique()))}
R = build_matrix(ui_train, user_index, item_index)
R_ext = sp.vstack([R, sp.csr_matrix((1, n_items))]).tocsr()          # last row = empty (cold user)
rows = np.array([user_index.get(u, len(user_index)) for u in eval_users])
R_eval = R_ext[rows]
cold = np.array([u not in user_index for u in eval_users])
seen = (R_eval > 0).toarray()

# user halves: hyper-parameters are chosen on A, reported on held-out users B
rng = np.random.default_rng(42)
perm = rng.permutation(len(eval_users))
A_idx, B_idx = perm[: len(perm) // 2], perm[len(perm) // 2:]
ALL_idx = np.arange(len(eval_users))

# ---- protocol statistics (used in the report) ----
tr_items = set(ui_train['tv_show_id'])
pool_set = set(int(s) for s in pool)
u_pos = {u: r for r, u in enumerate(eval_users)}
truth_flat = [(u, s) for u in eval_users for s in truths[u]]
print('PROTOCOL: train <', CUTOFF.date(), '| validation', CUTOFF.date(), '->', END.date())
print('eval users:', len(eval_users), '(of', len(sub_users), ') | cold users (no train history):', int(cold.sum()))
print('candidate pool size:', len(pool), '| pool items seen in train:', len(pool_set & tr_items))
print('truth items in pool: %.3f' % np.mean([s in pool_set for _, s in truth_flat]))
print('truth items already watched by that user in train (repeat viewing): %.3f'
      % np.mean([seen[u_pos[u], item_index[s]] for u, s in truth_flat]))
print('truth items never seen in train by anyone (cold items): %.3f' % np.mean([s not in tr_items for _, s in truth_flat]))
print('avg truth size per user: %.2f' % np.mean([len(v) for v in truths.values()]))

pop_all = popularity_scores(ui_train, item_index)
recent = train[train['ts'] >= CUTOFF - pd.Timedelta(days=28)]
pop_recent = popularity_scores(user_item(recent), item_index)
tie = 1e-6 * pop_all / max(pop_all.max(), 1)


def top5(S, exclude_seen):
    S = S + tie
    S = np.where(pool_mask[None, :], S, -np.inf)
    if exclude_seen:
        S = np.where(seen, -np.inf, S)
    idx = np.argsort(-S, axis=1)[:, :K]
    return {u: [int(item_ids[j]) for j in idx[r]] for r, u in enumerate(eval_users)}


def fill_cold(S, base):
    S = S.copy()
    S[cold] = base[None, :]
    return S


def score_subset(recs, idx):
    return evaluate(recs, {eval_users[i]: truths[eval_users[i]] for i in idx})


results, store = [], {}


def run(name, S, fit_s, pred_s, params='', exs=(False, True), verbose=True):
    for ex in exs:
        t = time.time()
        recs = top5(S, ex)
        row = {'method': name, 'params': params, 'exclude_seen': ex,
               'fit_s': round(fit_s, 1), 'pred_s': round(pred_s + time.time() - t, 2)}
        for tag, idx in (('A', A_idx), ('B', B_idx), ('All', ALL_idx)):
            for k, v in score_subset(recs, idx).items():
                row[f'{k}_{tag}'] = round(v, 4)
        results.append(row)
        store[(name, params, ex)] = recs
        if verbose:
            print(f"{name:10s} {params:30s} excl_seen={ex!s:5s} MAP@5 A={row['MAP@5_A']:.4f} "
                  f"B={row['MAP@5_B']:.4f} All={row['MAP@5_All']:.4f} HR@5={row['HR@5_All']:.3f}")


def best(name, ex=False):
    r = pd.DataFrame(results)
    r = r[(r['method'] == name) & (r['exclude_seen'] == ex)]
    return r.sort_values('MAP@5_A', ascending=False).iloc[0]


# ---- popularity baselines ----
run('pop_all', np.tile(pop_all, (len(eval_users), 1)), 0, 0)
run('pop_28d', np.tile(pop_recent, (len(eval_users), 1)), 0, 0)

# ---- own-history prior (repeat viewing), plain and recency-decayed ----
hist_scores = {}
for hl in (None, 14, 28, 56):
    t = time.time()
    H = history_matrix(train, CUTOFF, user_index, item_index, hl)
    He = sp.vstack([H, sp.csr_matrix((1, n_items))]).tocsr()[rows].toarray()
    p = f'half_life={hl}'
    hist_scores[p] = fill_cold(He, pop_all)
    run('hist', hist_scores[p], 0, time.time() - t, p)

# ---- content-based filtering ----
t = time.time()
V = item_features(items, item_index)
S_cbf = fill_cold(cbf_scores(R_eval, V), pop_all)
run('cbf', S_cbf, 0, time.time() - t, 'tfidf cat/genre/actor/dir/title/decade')

# ---- matrix factorization: implicit ALS ----
mf_scores = {}
for factors, alpha, reg in [(32, 1, 10), (64, 1, 30), (64, 0.5, 30), (64, 1, 10), (128, 1, 30)]:
    t = time.time()
    X, Y = als_fit(R, factors=factors, reg=reg, alpha=alpha, iters=8)
    fit_s = time.time() - t
    Xe = np.vstack([X, np.zeros((1, factors))])[rows]
    p = f'k={factors} alpha={alpha} reg={reg} it=8'
    mf_scores[('als', p)] = fill_cold(Xe @ Y.T, pop_all)
    run('als', mf_scores[('als', p)], fit_s, 0, p)

# ---- matrix factorization: truncated SVD of the weighted matrix (PureSVD) ----
for rank in (32, 64, 128):
    t = time.time()
    U, s, Vt = svds(R.astype(float), k=rank)
    fit_s = time.time() - t
    Ue = np.vstack([U, np.zeros((1, rank))])[rows]
    p = f'rank={rank}'
    mf_scores[('svd', p)] = fill_cold((Ue * s) @ Vt, pop_all)
    run('svd', mf_scores[('svd', p)], fit_s, 0, p)

# ---- hybrid: per-user z-scored blend; weights chosen on half A, reported on held-out half B ----
b_hist = best('hist')
b_als, b_svd = best('als'), best('svd')
b_mf = b_als if b_als['MAP@5_A'] >= b_svd['MAP@5_A'] else b_svd
S_hist = hist_scores[b_hist['params']]
S_mf = mf_scores[(b_mf['method'], b_mf['params'])]
print('\nselected on A -> hist:', b_hist['params'], '| MF:', b_mf['method'], b_mf['params'])


def z(S):
    P = np.where(pool_mask[None, :], S, np.nan)
    m, s = np.nanmean(P, 1, keepdims=True), np.nanstd(P, 1, keepdims=True) + 1e-9
    return (S - m) / s


Zh, Zm, Zc = z(S_hist), z(S_mf), z(S_cbf)
for wh, wm, wc in itertools.product((0, 0.5, 1, 2), (0, 0.5, 1), (0, 0.1, 0.25, 0.5)):
    if wh == wm == wc == 0:
        continue
    run('hybrid', wh * Zh + wm * Zm + wc * Zc, 0, 0, f'w_hist={wh} w_mf={wm} w_cbf={wc}',
        exs=(False,), verbose=False)

res = pd.DataFrame(results)
res.to_csv('results_validation.csv', index=False)
b_h = best('hybrid')
print('best hybrid on A:', b_h['params'], '| MAP@5 A=%.4f B=%.4f All=%.4f' % (b_h['MAP@5_A'], b_h['MAP@5_B'], b_h['MAP@5_All']))

pickle.dump({'recs': store, 'eval_users': eval_users, 'truths': truths, 'A_idx': A_idx, 'B_idx': B_idx,
             'best': {'hist': b_hist.to_dict(), 'mf': b_mf.to_dict(), 'hybrid': b_h.to_dict(),
                      'cbf': best('cbf').to_dict()}},
            open(f'{DATA}/val_recs.pkl', 'wb'))

print('\n=== SUMMARY (exclude_seen=False; params selected on A; MAP@5 on held-out B and on All) ===')
for m in ('pop_all', 'pop_28d', 'hist', 'cbf', 'als', 'svd', 'hybrid'):
    b = best(m)
    print(f"{m:8s} {b['params']:34s} B={b['MAP@5_B']:.4f} All={b['MAP@5_All']:.4f} "
          f"HR@5={b['HR@5_All']:.3f} P@5={b['P@5_All']:.3f} fit_s={b['fit_s']} pred_s={b['pred_s']}")
