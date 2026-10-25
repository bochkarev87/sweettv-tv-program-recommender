"""Popularity, content-based filtering and implicit-ALS matrix factorization."""
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def weight(sec):
    """Preference strength of a (user, show) pair: log of watched minutes."""
    return np.log1p(np.asarray(sec, dtype=float) / 60.0)


def build_matrix(ui, user_index, item_index):
    u = ui['user_id'].map(user_index).values
    i = ui['tv_show_id'].map(item_index).values
    ok = ~(pd.isna(u) | pd.isna(i))
    return sp.csr_matrix((weight(ui['sec'].values[ok]), (u[ok].astype(int), i[ok].astype(int))),
                         shape=(len(user_index), len(item_index)))


def history_matrix(train, cutoff, user_index, item_index, half_life_days):
    """Own-history prior: watch time decayed by age (half-life), log-scaled. half_life_days=None -> no decay."""
    t = train
    sec = t['overlap_sec'].values.astype(float)
    if half_life_days:
        age = (cutoff - t['ts']).dt.total_seconds().values / 86400.0
        sec = sec * 0.5 ** (age / half_life_days)
    d = pd.DataFrame({'user_id': t['user_id'].values, 'tv_show_id': t['tv_show_id'].values, 'sec': sec})
    return build_matrix(d.groupby(['user_id', 'tv_show_id'], as_index=False)['sec'].sum(), user_index, item_index)


# ---------------- popularity ----------------
def popularity_scores(ui, item_index):
    """Number of distinct users that watched each show."""
    s = np.zeros(len(item_index))
    c = ui.groupby('tv_show_id')['user_id'].nunique()
    for sid, n in c.items():
        s[item_index[sid]] = n
    return s


# ---------------- content-based ----------------
def _tokens(s, sep=','):
    if pd.isna(s):
        return []
    return [t.strip().lower() for t in str(s).split(sep) if t.strip()]


def item_features(items, item_index, weights=None, id_weight=0.0):
    """Weighted TF-IDF blocks: category, genres, actors, director, title words, decade.
    weights overrides block weights (0 drops a block); id_weight>0 adds a one-hot show-identity block."""
    it = items.set_index('tv_show_id').loc[list(item_index.keys())]
    fields = {
        'cat': ([_tokens(x) for x in it['tv_show_category']], 1.0),
        'gen': ([_tokens(a) + _tokens(b) + _tokens(c) for a, b, c in
                 zip(it['tv_show_genre_1'], it['tv_show_genre_2'], it['tv_show_genre_3'])], 1.0),
        'act': ([_tokens(x) for x in it['actors']], 0.5),
        'dir': ([_tokens(x) for x in it['director']], 0.5),
        'ttl': ([re.findall(r'[^\W\d_]{3,}', str(x).lower()) for x in it['tv_show_title']], 0.5),
        'dec': ([[str(y)[:3]] if pd.notna(y) and str(y)[:3].isdigit() else []
                 for y in it['year_of_production']], 0.3),
    }
    blocks = []
    for key, (toks, w) in fields.items():
        w = (weights or {}).get(key, w)
        if w == 0:
            continue
        v = TfidfVectorizer(analyzer=lambda d: d, min_df=1)
        try:
            m = v.fit_transform(toks)
        except ValueError:
            continue
        blocks.append(normalize(m) * w)
    if id_weight > 0:
        blocks.append(sp.identity(len(item_index), format='csr') * id_weight)
    return normalize(sp.hstack(blocks).tocsr())


def cbf_scores(R_users, V):
    """Cosine between the user profile (watch-weighted mean of item vectors) and every item."""
    P = normalize(R_users @ V)
    return (P @ V.T).toarray()


# ---------------- implicit ALS (Hu, Koren, Volinsky 2008) ----------------
def _solve_side(Rcsr, Y, alpha, reg):
    k = Y.shape[1]
    YtY = Y.T @ Y + reg * np.eye(k)
    X = np.zeros((Rcsr.shape[0], k))
    for u in range(Rcsr.shape[0]):
        a, b = Rcsr.indptr[u], Rcsr.indptr[u + 1]
        if a == b:
            continue
        idx = Rcsr.indices[a:b]
        cu = alpha * Rcsr.data[a:b]            # c_ui - 1
        Yi = Y[idx]
        A = YtY + (Yi.T * cu) @ Yi
        rhs = Yi.T @ (1.0 + cu)                # p_ui = 1 for observed pairs
        X[u] = np.linalg.solve(A, rhs)
    return X


def als_fit(R, factors=64, reg=1.0, alpha=10.0, iters=10, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.normal(0, 0.01, (R.shape[1], factors))
    Rt = R.T.tocsr()
    X = None
    for _ in range(iters):
        X = _solve_side(R, Y, alpha, reg)
        Y = _solve_side(Rt, X, alpha, reg)
    return X, Y
