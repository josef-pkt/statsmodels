import cStringIO as cio

import numpy as np
import pandas as pd


sio = cio.StringIO()
c = pd.read_hdf('kpss_critical_values.h5', 'c')
ct = pd.read_hdf('kpss_critical_values.h5', 'ct')

data = {'c': c, 'ct': ct}
for k, v in data.iteritems():
    n = v.shape[0]
    selected = np.zeros((n, 1), dtype=np.bool)
    selected[0] = True
    selected[-1] = True
    selected[v.index == 10.0] = True
    selected[v.index == 5.0] = True
    selected[v.index == 2.5] = True
    selected[v.index == 1.0] = True
    max_diff = 1.0
    while max_diff > 0.05:
        xp = np.squeeze(v[selected].values)
        yp = np.asarray(v[selected].index, dtype=np.float64)
        x = np.squeeze(v.values)
        y = np.asarray(v.index, dtype=np.float64)
        yi = np.interp(x, xp, yp)
        abs_diff = np.abs(y - yi)
        max_diff = np.max(abs_diff)
        if max_diff > 0.05:
            print selected.sum()
            print np.where(abs_diff == max_diff)
            selected[np.where(abs_diff == max_diff)] = True

    quantiles = list(np.squeeze(v[selected].index.values))
    critical_values = list(np.squeeze(v[selected].values))
    # Fix for first CV
    critical_values[0] = 0.0
    sio.write("kpss_critical_values['" + k + "'] = (")
    count = 0
    for c, q in zip(critical_values, quantiles):
        sio.write(
            '(' + '{0:0.1f}'.format(q) + ', ' + '{0:0.4f}'.format(c) + ')')
        count += 1
        if count % 3 == 0:
            sio.write(',\n                             ')
        else:
            sio.write(', ')
    sio.write(")\n")
    sio.write("kpss_critical_values['" + k + "'] = ")
    sio.write("np.array(kpss_critical_values['" + k + "'])")
    sio.write("\n")

sio.seek(0)
print sio.read()





kpss_critical_values = {}
kpss_critical_values['c'] = ((100.0, 0.0091), (99.5, 0.0218), (99.0, 0.0248),
                             (97.5, 0.0304), (95.5, 0.0354), (92.5, 0.0416),
                             (89.5, 0.0470), (73.5, 0.0726), (69.0, 0.0802),
                             (63.5, 0.0900), (58.5, 0.0999), (53.5, 0.1108),
                             (48.5, 0.1229), (43.0, 0.1376), (39.5, 0.1483),
                             (35.5, 0.1622), (31.5, 0.1780), (29.5, 0.1867),
                             (26.5, 0.2017), (24.5, 0.2127), (20.5, 0.2378),
                             (18.5, 0.2524), (16.0, 0.2738), (13.5, 0.2993),
                             (12.0, 0.3183), (9.5, 0.3549), (8.0, 0.3833),
                             (6.5, 0.4173), (4.5, 0.4793), (3.5, 0.5237),
                             (2.5, 0.5828), (1.5, 0.6734), (0.7, 0.8157),
                             (0.3, 0.9674), (0.1, 1.1692), )
kpss_critical_values['c'] = np.array(kpss_critical_values['c'])
kpss_critical_values['ct'] = ((100.0, 0.0077), (99.5, 0.0156), (99.0, 0.0173),
                             (98.0, 0.0194), (97.0, 0.0210), (95.5, 0.0229),
                             (94.0, 0.0245), (91.0, 0.0271), (88.0, 0.0295),
                             (85.5, 0.0313), (80.5, 0.0347), (67.0, 0.0434),
                             (60.0, 0.0480), (50.0, 0.0555), (46.5, 0.0584),
                             (40.0, 0.0643), (35.5, 0.0691), (32.0, 0.0731),
                             (29.0, 0.0769), (25.5, 0.0817), (21.0, 0.0894),
                             (18.5, 0.0943), (17.0, 0.0978), (15.5, 0.1015),
                             (13.0, 0.1088), (11.0, 0.1153), (8.5, 0.1257),
                             (7.0, 0.1337), (6.0, 0.1401), (4.0, 0.1570),
                             (2.5, 0.1770), (1.5, 0.1990), (0.7, 0.2350),
                             (0.3, 0.2738), (0.1, 0.3231), )
kpss_critical_values['ct'] = np.array(kpss_critical_values['ct'])