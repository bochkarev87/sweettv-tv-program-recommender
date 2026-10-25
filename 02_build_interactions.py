"""Join viewing logs with the TV schedule -> user-program interactions.

Matching rule: a log session (user, channel, [start, stop]) is matched to every
schedule entry on the SAME channel whose interval [p_start, p_start+duration)
overlaps the session. Overlap seconds are kept; pairs with overlap < MIN_OVERLAP
are dropped (channel zapping noise). Schedule rows with tv_show_id == 0 are
irrelevant shows and never become interactions.
"""
import os
import numpy as np
import pandas as pd

D = r'C:\Users\swaggaf0x\.cache\kagglehub\competitions\sweettv-tv-program-recommender'
OUT = 'data'
MIN_OVERLAP = 60      # seconds
MAX_PROGRAMS_PER_SESSION = 50
os.makedirs(OUT, exist_ok=True)

logs = pd.read_csv(f'{D}\\dataset11-30.csv', dtype={'user_id': 'string'})
logs['st'] = pd.to_datetime(logs['start_time'])
logs['en'] = pd.to_datetime(logs['stop_time'])
logs = logs[logs['en'] > logs['st']].reset_index(drop=True)

parts = []
for name in ['11-20', '21-30', '31-42']:
    s = pd.read_csv(f'{D}\\export_arh_{name}-final.csv', low_memory=False)
    s['week_file'] = name
    parts.append(s)
sched = pd.concat(parts, ignore_index=True)
sched['ps'] = pd.to_datetime(sched['start_time'], format='%d.%m.%Y %H:%M:%S')
sched['pe'] = sched['ps'] + pd.to_timedelta(sched['duration'], unit='s')
# files overlap around week boundaries -> keep one row per (channel, start)
sched = sched.drop_duplicates(['channel_id', 'ps'], keep='last')

# item metadata (all non-zero shows from all schedule files, incl. test weeks)
meta_cols = ['tv_show_id', 'tv_show_title', 'tv_show_category', 'tv_show_genre_1',
             'tv_show_genre_2', 'tv_show_genre_3', 'year_of_production', 'director', 'actors']
items = (sched.loc[sched['tv_show_id'] != 0, meta_cols]
         .drop_duplicates('tv_show_id').reset_index(drop=True))
items.to_pickle(f'{OUT}/items.pkl')
# airings of non-zero shows (used later for candidate pools)
air = sched.loc[sched['tv_show_id'] != 0, ['channel_id', 'tv_show_id', 'ps', 'pe', 'week_file']]
air.to_pickle(f'{OUT}/airings.pkl')
print('items:', len(items), ' airings:', len(air))

# ---- vectorised interval join per channel ----
rows = []
unmatched_channels = set()
for ch, g in logs.groupby('vsetv_id'):
    p = air[air['channel_id'] == ch].sort_values('ps')
    if p.empty:
        unmatched_channels.add(ch)
        continue
    ps = p['ps'].values.astype('datetime64[s]').astype('int64')
    pe = p['pe'].values.astype('datetime64[s]').astype('int64')
    pe_max = np.maximum.accumulate(pe)
    sid = p['tv_show_id'].values
    st = g['st'].values.astype('datetime64[s]').astype('int64')
    en = g['en'].values.astype('datetime64[s]').astype('int64')
    lo = np.searchsorted(pe_max, st, side='right')   # first program ending after session start
    hi = np.searchsorted(ps, en, side='left')        # programs starting before session end
    cnt = np.clip(hi - lo, 0, MAX_PROGRAMS_PER_SESSION)
    if cnt.sum() == 0:
        continue
    sess = np.repeat(np.arange(len(g)), cnt)
    offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    pidx = np.repeat(lo, cnt) + offs
    ov = np.minimum(en[sess], pe[pidx]) - np.maximum(st[sess], ps[pidx])
    keep = ov >= MIN_OVERLAP
    rows.append(pd.DataFrame({
        'user_id': g['user_id'].values[sess][keep],
        'tv_show_id': sid[pidx][keep],
        'channel_id': ch,
        'ts': pd.to_datetime(st[sess][keep], unit='s'),
        'overlap_sec': ov[keep],
    }))

inter = pd.concat(rows, ignore_index=True)
inter.to_pickle(f'{OUT}/interactions.pkl')

total_watch = (logs['en'] - logs['st']).dt.total_seconds().sum()
print('log sessions:', len(logs), ' users:', logs['user_id'].nunique())
print('channels in logs with no non-zero schedule entries:', sorted(unmatched_channels))
print('interactions (session x show):', len(inter),
      ' users:', inter['user_id'].nunique(), ' shows:', inter['tv_show_id'].nunique())
print('share of watch time matched to relevant shows: %.3f' % (inter['overlap_sec'].sum() / total_watch))
print('overlap_sec describe:\n', inter['overlap_sec'].describe())
print('ts range:', inter['ts'].min(), '->', inter['ts'].max())
