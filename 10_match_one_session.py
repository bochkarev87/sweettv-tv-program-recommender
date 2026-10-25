"""Helper for the manual check: show one raw log session, the schedule rows around it, and what the pipeline stored.
Compute the overlap seconds by hand, then compare with the 'pipeline overlap_sec' line."""
import pandas as pd
from common import D

USER = '18204348906365533863'          # user A of the report
logs = pd.read_csv(rf'{D}\dataset11-30.csv', dtype={'user_id': 'string'})
logs = logs[logs['user_id'] == USER].copy()
logs['st'], logs['en'] = pd.to_datetime(logs['start_time']), pd.to_datetime(logs['stop_time'])
logs['dur_min'] = (logs['en'] - logs['st']).dt.total_seconds() / 60
inter = pd.read_pickle('data/interactions.pkl')
inter = inter[inter['user_id'] == USER]
cand = logs[(logs['dur_min'] > 25) & (logs['dur_min'] < 40) & (logs['st'] > '2020-04-01')]
for _, s in cand.iterrows():
    m = inter[(inter['channel_id'] == s['vsetv_id']) & (inter['ts'] == s['st'])]
    if len(m) >= 2:          # a session that spans two programs is the most instructive
        break
print('LOG SESSION: channel', s['vsetv_id'], '| start', s['st'], '| stop', s['en'])
sched = pd.concat([pd.read_csv(rf'{D}\export_arh_{n}-final.csv', low_memory=False) for n in ('11-20', '21-30')])
sched['ps'] = pd.to_datetime(sched['start_time'], format='%d.%m.%Y %H:%M:%S')
sched['pe'] = sched['ps'] + pd.to_timedelta(sched['duration'], unit='s')
w = sched[(sched['channel_id'] == s['vsetv_id']) & (sched['pe'] > s['st'] - pd.Timedelta('30min')) &
          (sched['ps'] < s['en'] + pd.Timedelta('30min'))].drop_duplicates(['channel_id', 'ps']).sort_values('ps')
print('\nSCHEDULE rows on this channel around the session (tv_show_id=0 means irrelevant):')
print(w[['ps', 'pe', 'duration', 'tv_show_id', 'tv_show_title']].to_string(index=False))
print('\nPIPELINE stored for this session (user, show, overlap_sec):')
print(m[['tv_show_id', 'overlap_sec']].to_string(index=False))
