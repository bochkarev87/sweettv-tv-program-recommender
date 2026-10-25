import pandas as pd

D = r'C:\Users\swaggaf0x\.cache\kagglehub\competitions\sweettv-tv-program-recommender'

# --- логи ---
logs = pd.read_csv(f'{D}\\dataset11-30.csv', dtype={'user_id': 'string'})
print('LOGS', logs.shape)
print(logs.dtypes)
logs['start_time'] = pd.to_datetime(logs['start_time'], format='%Y-%m-%d %H:%M:%S')
logs['stop_time'] = pd.to_datetime(logs['stop_time'], format='%Y-%m-%d %H:%M:%S')
print('users:', logs['user_id'].nunique(), ' channels:', logs['vsetv_id'].nunique())
print('log range:', logs['start_time'].min(), '->', logs['start_time'].max())

# --- расписания ---
sched = {}
for name in ['11-20', '21-30', '31-42']:
    s = pd.read_csv(f'{D}\\export_arh_{name}-final.csv', dtype={'tv_show_id': 'int64'})
    s['start_time'] = pd.to_datetime(s['start_time'], format='%d.%m.%Y %H:%M:%S')
    sched[name] = s
    print(f'SCHED {name}', s.shape,
          'range:', s['start_time'].min(), '->', s['start_time'].max(),
          '| share tv_show_id=0: %.3f' % (s['tv_show_id'] == 0).mean(),
          '| unique shows (non-zero):', s.loc[s['tv_show_id'] != 0, 'tv_show_id'].nunique(),
          '| channels:', s['channel_id'].nunique())

# --- submission ---
sub = pd.read_csv(f'{D}\\submission.csv', dtype={'user_id': 'string'})
print('SUB users:', len(sub), ' in logs:', sub['user_id'].isin(logs['user_id']).sum())

# --- пересечение каналов логов и расписания ---
ch_logs = set(logs['vsetv_id'].unique())
ch_sched = set(sched['11-20']['channel_id'].unique()) | set(sched['21-30']['channel_id'].unique())
print('log channels covered by schedule:', len(ch_logs & ch_sched), 'of', len(ch_logs))
