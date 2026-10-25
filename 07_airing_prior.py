"""IMPROVE step 2: add a schedule-based item prior (how often a show is on air in the target window).
The schedule of the target window is known in advance (provided metadata), so this is not leakage."""
import os
import re
import pickle
import itertools
exec(open('03_experiments.py').read().split('# ---- popularity baselines ----')[0])

TAG = os.environ.get('SWEET_TAG', '')
sel = pickle.load(open(f'{DATA}/val_recs.pkl', 'rb'))['best']
hl = float(re.search(r'half_life=(\w+)', sel['hist']['params']).group(1))
rank = int(re.search(r'rank=(\d+)', sel['mf']['params']).group(1))

H = history_matrix(train, CUTOFF, user_index, item_index, hl)
S_hist = fill_cold(sp.vstack([H, sp.csr_matrix((1, n_items))]).tocsr()[rows].toarray(), pop_all)
U, s, Vt = svds(R.astype(float), k=rank)
S_mf = fill_cold((np.vstack([U, np.zeros((1, rank))])[rows] * s) @ Vt, pop_all)

win = air[(air['ps'] >= CUTOFF) & (air['ps'] < END)].copy()
win['hours'] = (win['pe'] - win['ps']).dt.total_seconds() / 3600
n_air = np.zeros(n_items)
hours = np.zeros(n_items)
for sid, g in win.groupby('tv_show_id'):
    n_air[item_index[int(sid)]] = len(g)
    hours[item_index[int(sid)]] = g['hours'].sum()
PRIORS = {'log_n_airings': np.log1p(n_air), 'log_air_hours': np.log1p(hours)}
print('pool shows: median #airings = %.0f, max = %.0f' % (np.median(n_air[pool_mask]), n_air.max()))


def z(S):
    P = np.where(pool_mask[None, :], S, np.nan)
    return (S - np.nanmean(P, 1, keepdims=True)) / (np.nanstd(P, 1, keepdims=True) + 1e-9)


Zh, Zm = z(S_hist), z(S_mf)
for pname, pv in PRIORS.items():
    run('prior_only', np.tile(pv, (len(eval_users), 1)), 0, 0, pname, exs=(False,))
    Zp = z(np.tile(pv, (len(eval_users), 1)))
    for wh, wm, wp in itertools.product((1, 2), (0, 0.5), (0, 0.25, 0.5, 1, 2)):
        run('hybrid_air', wh * Zh + wm * Zm + wp * Zp, 0, 0, f'{pname} w_hist={wh} w_mf={wm} w_prior={wp}',
            exs=(False,), verbose=False)

r = pd.DataFrame(results)
r.to_csv(f'results_airing_prior{TAG}.csv', index=False)
print(r[r['method'] == 'prior_only'][['params', 'MAP@5_All']].to_string(index=False))
h = r[r['method'] == 'hybrid_air'].sort_values('MAP@5_A', ascending=False)
print('\ntop-6 by half A:')
print(h[['params', 'MAP@5_A', 'MAP@5_B', 'MAP@5_All', 'HR@5_All']].head(6).to_string(index=False))
base = h[h['params'].str.contains('w_prior=0$')].sort_values('MAP@5_A', ascending=False).iloc[0]
print('\nbaseline without prior (best on A): %s | B=%.4f All=%.4f' % (base['params'], base['MAP@5_B'], base['MAP@5_All']))
b = h.iloc[0]
m = re.search(r'(\w+) w_hist=([\d.]+) w_mf=([\d.]+) w_prior=([\d.]+)', b['params'])
pickle.dump({'prior': m.group(1), 'w_hist': float(m.group(2)), 'w_mf': float(m.group(3)), 'w_prior': float(m.group(4)),
             'val': b.to_dict(), 'baseline': base.to_dict()}, open(f'{DATA}/final_cfg_air{TAG}.pkl', 'wb'))
