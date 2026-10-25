"""Sanity / leakage checks that back the claims in the report."""
import numpy as np
import pandas as pd
from common import *

# 1) AP@5 reproduces the worked example of the course slides (slide 5): 0.917 and 0.533
x, y = apk(list('ABDCE'), list('ABC')), apk(list('DAEBC'), list('ABC'))
assert abs(x - 0.9167) < 1e-3 and abs(y - 0.5333) < 1e-3, (x, y)
print('AP@5 slide example: X=%.3f (slide 0.917), Y=%.3f (slide 0.533)  OK' % (x, y))
# 2) hand-checkable case: hits at ranks 1,3,4,5 with 5 relevant -> (1/1+2/3+3/4+4/5)/5
v = apk([1, 9, 2, 3, 4], [1, 2, 3, 4, 5])
assert abs(v - (1 + 2 / 3 + 3 / 4 + 4 / 5) / 5) < 1e-12
print('AP@5 hits at ranks 1,3,4,5 of 5 relevant = %.4f  OK' % v)

inter, items, air, sub = load()
train = inter[inter['ts'] < CUTOFF]
val = inter[(inter['ts'] >= CUTOFF) & (inter['ts'] < END)]
assert train['ts'].max() < CUTOFF <= val['ts'].min()
print('train ts max', train['ts'].max(), '< cutoff', CUTOFF.date(), '<= validation ts min', val['ts'].min(), ' OK')
assert (inter['tv_show_id'] != 0).all() and (inter['overlap_sec'] >= 60).all()
print('interactions: no tv_show_id=0, min overlap', inter['overlap_sec'].min(), 's  OK')

pool = set(candidate_pool(air, CUTOFF, END).tolist())
tr = make_truth(val, sub['user_id'].tolist())
assert all(s in pool for v_ in tr.values() for s in v_)
print('all validation targets are inside the candidate pool (%d users)  OK' % len(tr))

for f in ('submission.csv', 'submission_v2_airprior.csv'):
    s = pd.read_csv(f, dtype=str)
    t = pd.read_csv(f'{D}\submission.csv', dtype=str)
    test_pool = set(candidate_pool(air, TEST_START, pd.Timestamp('2100-01-01')).tolist())
    assert s['user_id'].tolist() == t['user_id'].tolist() and len(s) == 1260
    assert all(len(set(map(int, r.split()))) == 5 and set(map(int, r.split())) <= test_pool for r in s['tv_show_id'])
    print(f, ': 1260 users, 5 distinct programs each, all in the weeks 31-42 schedule  OK')
