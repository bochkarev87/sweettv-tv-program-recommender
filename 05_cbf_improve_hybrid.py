"""IMPROVE step: ablate CBF feature blocks (+ show-identity feature), then re-tune the hybrid.
Reuses the exact split / pool / protocol of 03_experiments.py (its setup block is exec'd)."""
import re
import pickle
import itertools
exec(open('03_experiments.py').read().split('# ---- popularity baselines ----')[0])

sel = pickle.load(open(f'{DATA}/val_recs.pkl', 'rb'))['best']
hl = float(re.search(r'half_life=(\w+)', sel['hist']['params']).group(1))
rank = int(re.search(r'rank=(\d+)', sel['mf']['params']).group(1))
assert sel['mf']['method'] == 'svd'

H = history_matrix(train, CUTOFF, user_index, item_index, hl)
S_hist = fill_cold(sp.vstack([H, sp.csr_matrix((1, n_items))]).tocsr()[rows].toarray(), pop_all)
U, s, Vt = svds(R.astype(float), k=rank)
S_mf = fill_cold((np.vstack([U, np.zeros((1, rank))])[rows] * s) @ Vt, pop_all)

CBF_CFGS = {
    'cat+genre': ({'act': 0, 'dir': 0, 'ttl': 0, 'dec': 0}, 0.0),
    'cat+genre+title': ({'act': 0, 'dir': 0, 'dec': 0}, 0.0),
    'all blocks (03 default)': ({}, 0.0),
    'cat+genre + id1': ({'act': 0, 'dir': 0, 'ttl': 0, 'dec': 0}, 1.0),
    'all + id0.5': ({}, 0.5),
    'all + id1': ({}, 1.0),
    'all + id2': ({}, 2.0),
}
cbf_S = {}
for name, (w, idw) in CBF_CFGS.items():
    t = time.time()
    Vf = item_features(items, item_index, w, idw)
    cbf_S[name] = fill_cold(cbf_scores(R_eval, Vf), pop_all)
    run('cbf2', cbf_S[name], 0, time.time() - t, name, exs=(False,))

r = pd.DataFrame(results)
bc = r.sort_values('MAP@5_A', ascending=False).iloc[0]
print('\nbest CBF on A:', bc['params'], '| B=%.4f All=%.4f' % (bc['MAP@5_B'], bc['MAP@5_All']))
S_cbf2 = cbf_S[bc['params']]

def z(S):
    P = np.where(pool_mask[None, :], S, np.nan)
    return (S - np.nanmean(P, 1, keepdims=True)) / (np.nanstd(P, 1, keepdims=True) + 1e-9)


Zh, Zm, Zc = z(S_hist), z(S_mf), z(S_cbf2)
n0 = len(results)
for wh, wm, wc in itertools.product((0, 1, 2), (0, 0.5, 1), (0, 0.25, 0.5, 1)):
    if wh == wm == wc == 0:
        continue
    run('hybrid2', wh * Zh + wm * Zm + wc * Zc, 0, 0, f'w_hist={wh} w_mf={wm} w_cbf={wc}', exs=(False,), verbose=False)

r = pd.DataFrame(results)
r.to_csv('results_cbf_hybrid2.csv', index=False)
h = r[r['method'] == 'hybrid2'].sort_values('MAP@5_A', ascending=False)
print('\ntop-5 hybrid2 configs by A:')
print(h[['params', 'MAP@5_A', 'MAP@5_B', 'MAP@5_All', 'HR@5_All', 'P@5_All']].head(5).to_string(index=False))
bh = h.iloc[0]
wh, wm, wc = [float(x) for x in re.findall(r'w_hist=([\d.]+) w_mf=([\d.]+) w_cbf=([\d.]+)', bh['params'])[0]]
pickle.dump({'hist_half_life': hl, 'svd_rank': rank, 'cbf_name': bc['params'],
             'cbf_weights': CBF_CFGS[bc['params']][0], 'cbf_id_weight': CBF_CFGS[bc['params']][1],
             'w_hist': wh, 'w_mf': wm, 'w_cbf': wc, 'val_hybrid': bh.to_dict(), 'val_cbf': bc.to_dict()},
            open(f'{DATA}/final_cfg.pkl', 'wb'))
cur = pickle.load(open(f'{DATA}/val_recs.pkl', 'rb'))
cur['recs'].update(store)
cur['best2'] = {'cbf2': bc.to_dict(), 'hybrid2': bh.to_dict()}
pickle.dump(cur, open(f'{DATA}/val_recs.pkl', 'wb'))
print('\nselected:', {'cbf': bc['params'], 'w_hist': wh, 'w_mf': wm, 'w_cbf': wc})
