"""Sanity check of the CBF pipeline: with a very large show-identity weight CBF must approach the
own-history baseline; plus a per-user diagnosis of why plain CBF fails."""
exec(open('03_experiments.py').read().split('# ---- popularity baselines ----')[0])

for idw in (0, 2, 5, 20, 100):
    V = item_features(items, item_index, {}, idw)
    S = fill_cold(cbf_scores(R_eval, V), pop_all)
    run('cbf_id', S, 0, 0, f'id_weight={idw}', exs=(False,))

V = item_features(items, item_index, {}, 2.0)
r = 5
P = normalize(R_eval[r] @ V)
sc = (P @ V.T).toarray().ravel()
top = np.argsort(-np.where(pool_mask, sc, -np.inf))[:5]
it = items.set_index('tv_show_id')
print('\nuser', eval_users[r], '| distinct shows watched in train:', int((R_eval[r] > 0).sum()))
print('CBF top-5 (id, category, already watched):',
      [(int(item_ids[j]), it.loc[item_ids[j], 'tv_show_category'], bool(seen[r, j])) for j in top])
print('truth (id, category):', [(t, it.loc[t, 'tv_show_category']) for t in truths[eval_users[r]]])
tj = [item_index[t] for t in truths[eval_users[r]]]
print('cosine score of truth items:', np.round(sc[tj], 3), '| cosine score of CBF top-5:', np.round(sc[top], 3))
