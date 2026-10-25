"""Two-user case study on the validation split: history, truth, and every method's Top-5."""
import pickle
import numpy as np
import pandas as pd
from common import *

inter, items, air, sub = load()
val_pkl = pickle.load(open(f'{DATA}/val_recs.pkl', 'rb'))
recs, eval_users, truths = val_pkl['recs'], val_pkl['eval_users'], val_pkl['truths']
res = pd.read_csv('results_validation.csv')
als_p = res[(res['method'] == 'als') & (~res['exclude_seen'])].sort_values('MAP@5_A', ascending=False).iloc[0]['params']
sel = val_pkl['best']
METHODS = {
    'Popularity': ('pop_all', ''),
    'Own history': ('hist', sel['hist']['params']),
    'CBF': ('cbf', sel['cbf']['params']),
    'ALS (MF)': ('als', als_p),
    'SVD (MF)': ('svd', sel['mf']['params']),
    'Hybrid': ('hybrid', sel['hybrid']['params']),
}
rec_by = {}
for label, (m, p) in METHODS.items():
    key = [k for k in recs if k[0] == m and (k[1] == p or (m == 'pop_all' and k[1] in ('', None) or (isinstance(k[1], float)))) and k[2] is False]
    rec_by[label] = recs[(m, p if p != '' else key[0][1], False)] if (m, p, False) in recs else recs[key[0]]

train = inter[inter['ts'] < CUTOFF]
ui = user_item(train)
n_shows = ui.groupby('user_id')['tv_show_id'].nunique()
cand = n_shows.reindex(eval_users).dropna()
med, p10 = cand.median(), cand.quantile(0.10)
u_typ = (cand - med).abs().sort_values().index[0]
u_light = (cand - p10).abs().sort_values().index[0]
print(f'train distinct shows per eval user: median={med:.0f}, 10th pct={p10:.0f}')
print('selected: typical user (closest to median activity) =', u_typ, f'({int(cand[u_typ])} shows);',
      'light user (closest to 10th percentile) =', u_light, f'({int(cand[u_light])} shows)')

meta = items.drop_duplicates('tv_show_id').set_index('tv_show_id')


def desc(sid):
    t = str(meta.loc[sid, 'tv_show_title'])[:48]
    return f"{sid:>9} | {meta.loc[sid, 'tv_show_category']:<14} | {t}"


for u in (u_typ, u_light):
    print('\n' + '=' * 100)
    h = ui[ui['user_id'] == u].sort_values('sec', ascending=False)
    cats = h.merge(meta[['tv_show_category']], left_on='tv_show_id', right_index=True).groupby('tv_show_category')['sec'].sum()
    print(f'USER {u}: {len(h)} distinct shows in train, total {h["sec"].sum() / 3600:.1f} h')
    print('watch-time share by category:', {k: round(v / cats.sum(), 2) for k, v in cats.sort_values(ascending=False).head(4).items()})
    print('TRAIN top-8 by watch time:')
    for _, r in h.head(8).iterrows():
        print('  ', desc(int(r['tv_show_id'])), f"| {r['sec'] / 60:.0f} min")
    tr = truths[u]
    print('VALIDATION truth (top-5 by watch time):')
    for s in tr:
        print('  ', desc(s), '| seen in train' if s in set(h['tv_show_id']) else '| NEW (not in user train history)')
    for label, rc in rec_by.items():
        r = rc[u]
        hits = [int(x in tr) for x in r]
        print(f'\n{label}: AP@5={apk(r, tr):.3f}, hits={sum(hits)}/5')
        for s, hit in zip(r, hits):
            print('  ', desc(s), '| HIT' if hit else '|', '| in train history' if s in set(h['tv_show_id']) else '| not in history')

# hand-check of AP@5 for the typical user, hybrid (to be reproduced by hand in the report)
u = u_typ
r, tr = rec_by['Hybrid'][u], truths[u]
print('\n' + '=' * 100)
print('AP@5 check, typical user, Hybrid: ranks 1..5 hit flags =', [int(x in tr) for x in r])
hits, terms = 0, []
for i, x in enumerate(r, 1):
    if x in tr:
        hits += 1
        terms.append(f'{hits}/{i}')
print('AP@5 = (' + ' + '.join(terms) + f') / min(|truth|={len(tr)}, 5) =', round(apk(r, tr), 4))
