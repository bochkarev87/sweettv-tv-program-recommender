"""Build report.html from result files (all numbers are read from the experiment outputs), then print to PDF."""
import html
import os
import pickle
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT0 = os.path.abspath('..')
ROOT = ROOT0; sys.path.insert(0, ROOT)
os.chdir(ROOT)
from common import user_item, CUTOFF, END, apk  # noqa: E402

E = html.escape
res = pd.read_csv('results_validation.csv')
res2 = pd.read_csv('results_cbf_hybrid2.csv')
air8 = pd.read_csv('results_airing_prior.csv')
air12 = pd.read_csv('results_airing_prior_h12w.csv')
val = pickle.load(open('data/val_recs.pkl', 'rb'))
vair = pickle.load(open('data/val_recs_air.pkl', 'rb'))
items = pd.read_pickle('data/items.pkl').drop_duplicates('tv_show_id').set_index('tv_show_id')


def row(df, method, params=None, ex=False):
    d = df[(df['method'] == method) & (df['exclude_seen'] == ex)]
    if params is not None:
        d = d[d['params'] == params]
    else:
        d = d.sort_values('MAP@5_A', ascending=False)
    return d.iloc[0]


r_pop = row(res, 'pop_all')
r_pop28 = row(res, 'pop_28d')
r_hist = row(res, 'hist', val['best']['hist']['params'])
r_cbf = row(res, 'cbf')
r_cbf2 = row(res2, 'cbf2', 'all + id2')
r_als = row(res, 'als')
r_svd = row(res, 'svd', val['best']['mf']['params'])
r_h1 = row(res, 'hybrid', val['best']['hybrid']['params'])
r_h2 = row(air8, 'hybrid_air', vair['params'])
r_h2base = row(air8, 'hybrid_air', vair['baseline_params'])
r_h12_2 = air12[(air12['method'] == 'hybrid_air')].sort_values('MAP@5_A', ascending=False).iloc[0]
r_h12_0 = air12[(air12['method'] == 'hybrid_air') & (air12['params'].str.contains('w_prior=0$'))] \
    .sort_values('MAP@5_A', ascending=False).iloc[0]
ex_pop, ex_hist, ex_svd = row(res, 'pop_all', ex=True), row(res, 'hist', val['best']['hist']['params'], True), \
    row(res, 'svd', val['best']['mf']['params'], True)


def f(x, d=4):
    return f'{float(x):.{d}f}'


def trow(name, cfg, r, note=''):
    return (f"<tr><td>{name}</td><td>{cfg}</td><td>{f(r['MAP@5_B'])}</td><td>{f(r['MAP@5_All'])}</td>"
            f"<td>{f(r['HR@5_All'], 3)}</td><td>{f(r['P@5_All'], 3)}</td><td>{r['fit_s']}</td>"
            f"<td>{r['pred_s']}</td></tr>")


main_table = ''.join([
    trow('Popularity (#users)', 'all train', r_pop),
    trow('Popularity, last 28 d', '28-day window', r_pop28),
    trow('Own-history prior (ref.)', E(r_hist['params']).replace('half_life', 'half-life') + ' d', r_hist),
    trow('<b>CBF</b> (required method 1)', 'TF-IDF: cat/genre/actor/dir/title/decade', r_cbf),
    trow('CBF + show-identity feature', 'identity weight 2', r_cbf2),
    trow('<b>MF: ALS</b> (required method 2a)', E(r_als['params']), r_als),
    trow('<b>MF: PureSVD</b> (required method 2b)', E(r_svd['params']), r_svd),
    trow('Hybrid v1 (history+SVD)', 'w=2 / 0.5 (z-scored)', r_h1),
    trow('<b>Hybrid v2 = v1 + airing prior (final)</b>', 'prior w=1, log #airings', r_h2),
])


# ---------- two-user case study ----------
inter = pd.read_pickle('data/interactions.pkl')
train = inter[inter['ts'] < CUTOFF]
ui = user_item(train)
nsh = ui.groupby('user_id')['tv_show_id'].nunique().reindex(val['eval_users']).dropna()
med, p10 = nsh.median(), nsh.quantile(0.10)
u_typ = (nsh - med).abs().sort_values().index[0]
u_light = (nsh - p10).abs().sort_values().index[0]
rec = val['recs']
METH = [
    ('Popularity', rec[('pop_all', '', False)]),
    ('CBF', rec[('cbf', val['best']['cbf']['params'], False)]),
    ('ALS', rec[('als', r_als['params'], False)]),
    ('SVD', rec[('svd', r_svd['params'], False)]),
    ('Hybrid v1', rec[('hybrid', r_h1['params'], False)]),
    ('Hybrid v2 (final)', vair['final']),
]


def short(sid, n=24):
    t = str(items.loc[sid, 'tv_show_title']).strip(" '.")
    c = str(items.loc[sid, 'tv_show_category'])[:6]
    return f"{sid} {E(t[:n])}{'…' if len(t) > n else ''} <i>({E(c)})</i>"


def user_block(u, label):
    tr = val['truths'][u]
    h = ui[ui['user_id'] == u].sort_values('sec', ascending=False)
    hs = set(h['tv_show_id'])
    hist5 = '; '.join(short(int(s), 20) for s in h['tv_show_id'].head(5))
    truth_s = '; '.join(short(s, 20) + ('' if s in hs else ' <b>[new]</b>') for s in tr)
    head = ''.join(f"<th>{n}<br>AP@5={apk(rc[u], tr):.3f}</th>" for n, rc in METH)
    body = ''
    for i in range(5):
        cells = ''
        for n, rc in METH:
            s = rc[u][i]
            cls = ' class="hit"' if s in tr else ''
            cells += f"<td{cls}>{short(s, 18)}{' ✓' if s in tr else ''}</td>"
        body += f"<tr><td>{i + 1}</td>{cells}</tr>"
    return (f"<p class='small'><b>{label}</b>: user …{str(u)[-6:]}, {len(h)} distinct shows / "
            f"{h['sec'].sum() / 3600:.0f} h in training. <b>Train top-5 by watch time:</b> {hist5}. "
            f"<b>Validation truth:</b> {truth_s}.</p>"
            f"<table class='t small'><tr><th>#</th>{head}</tr>{body}</table>")


case_html = user_block(u_typ, 'User A (typical: median activity)') + user_block(u_light, 'User B (light: 10th-percentile activity)')
def hits(rc, u):
    return len(set(rc[u]) & set(val['truths'][u]))


ap_typ_h2 = apk(vair['final'][u_typ], val['truths'][u_typ])
rec_h1 = dict(METH)['Hybrid v1']
rec_svd = dict(METH)['SVD']

shot = 'report/kaggle_submissions.webp'
tmpl = open('report/template.html', encoding='utf-8').read()
subs = {
    'MAIN_TABLE': main_table, 'CASE_HTML': case_html,
    'POP_ALL': f(r_pop['MAP@5_All'], 3), 'HIST_ALL': f(r_hist['MAP@5_All'], 3), 'CBF_ALL': f(r_cbf['MAP@5_All'], 4),
    'CBF2_ALL': f(r_cbf2['MAP@5_All'], 4), 'ALS_ALL': f(r_als['MAP@5_All'], 3), 'SVD_ALL': f(r_svd['MAP@5_All'], 3),
    'H1_ALL': f(r_h1['MAP@5_All'], 4), 'H2_ALL': f(r_h2['MAP@5_All'], 4), 'H1_B': f(r_h1['MAP@5_B'], 4),
    'H2_B': f(r_h2['MAP@5_B'], 4), 'H2BASE_B': f(r_h2base['MAP@5_B'], 4), 'H2BASE_ALL': f(r_h2base['MAP@5_All'], 4),
    'H12_0_B': f(r_h12_0['MAP@5_B'], 4), 'H12_2_B': f(r_h12_2['MAP@5_B'], 4),
    'H12_0_ALL': f(r_h12_0['MAP@5_All'], 4), 'H12_2_ALL': f(r_h12_2['MAP@5_All'], 4),
    'EXPOP': f(ex_pop['MAP@5_All'], 4), 'EXHIST': f(ex_hist['MAP@5_All'], 4), 'EXSVD': f(ex_svd['MAP@5_All'], 4),
    'ALS_FIT': str(r_als['fit_s']), 'SVD_FIT': str(r_svd['fit_s']),
    'HITS_TYP_H2': str(hits(vair['final'], u_typ)), 'HITS_TYP_H1': str(hits(rec_h1, u_typ)),
    'AP_TYP_H1': f"{apk(rec_h1[u_typ], val['truths'][u_typ]):.3f}",
    'AP_LIGHT_H1': f"{apk(rec_h1[u_light], val['truths'][u_light]):.3f}",
    'AP_LIGHT_H2': f"{apk(vair['final'][u_light], val['truths'][u_light]):.3f}",
    'AP_LIGHT_SVD': f"{apk(rec_svd[u_light], val['truths'][u_light]):.3f}",
    'N_MED': f'{med:.0f}', 'N_P10': f'{p10:.0f}', 'AP_TYP_H2': f'{ap_typ_h2:.3f}', 'SHOT': shot,
}
out = tmpl
for k, v in subs.items():
    out = out.replace('{{' + k + '}}', v)
assert '{{' not in out, 'unfilled placeholder'
open('report/report.html', 'w', encoding='utf-8').write(out)
print('report.html written;', 'users:', u_typ, u_light)

chrome = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
pdf = os.path.abspath('report/report.pdf')
subprocess.run([chrome, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={pdf}',
                'file:///' + os.path.abspath('report/report.html').replace('\\', '/')], check=True, timeout=120)
from pypdf import PdfReader  # noqa: E402
print('PDF pages (incl. references):', len(PdfReader(pdf).pages))
