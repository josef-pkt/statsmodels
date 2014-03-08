"""
Calculates quantiles of the KPSS test statistic for both the constant
and constant plus trend scenarios.
"""
from __future__ import division
import os
from numpy.random import RandomState
import numpy as np
import pandas as pd
from statsmodels.tsa.tsatools import add_trend


def simulate_kpss(nobs, B, deterministic='c', rng=None):
    if rng is None:
        rng = RandomState()
        rng.seed(0)

    standard_normal = rng.standard_normal

    e = standard_normal((nobs, B))
    z = np.ones((nobs, 1))
    if deterministic == 'ct':
        z = add_trend(z, trend='t')
    zinv = np.linalg.pinv(z)
    trend_coef = zinv.dot(e)
    resid = e - z.dot(trend_coef)
    s = np.cumsum(resid, axis=0)
    lam = np.mean(resid ** 2.0, axis=0)
    kpss = 1 / (nobs ** 2.0) * np.sum(s ** 2.0, axis=0) / lam
    return kpss


def block_simulate_kpss(nobs, b, deterministic='c', max_memory=250):
    rng = RandomState()
    rng.seed(0)
    memory = max_memory * 2 ** 20
    b_max_memory = memory // 8 // nobs
    b_max_memory = max(b_max_memory, 1)
    remaining = b
    results = np.zeros(b)
    while remaining > 0:
        b_eff = min(remaining, b_max_memory)
        completed = b - remaining
        results[completed:completed + b_eff] = \
            simulate_kpss(nobs, b_eff, deterministic=deterministic, rng=rng)
        remaining -= b_max_memory

    return results


if __name__ == '__main__':
    import datetime as dt
    import cStringIO as cio

    nobs = 2000
    B = 10 * 10e6

    percentiles = np.concatenate((np.arange(0.0, 99.0, 0.5),
                                  np.arange(99.0, 100.0, 0.1)))

    critical_values = 100 - percentiles
    critical_values_string = map(lambda x: '{0:0.1f}'.format(x), critical_values)
    sio = cio.StringIO()
    sio.write("kpss_critical_values = {}\n")

    hdf_filename = 'kpss_critical_values.h5'
    try:
        os.remove(hdf_filename)
    except OSError:
        pass

    for d in ('c', 'ct'):
        now = dt.datetime.now()
        kpss = block_simulate_kpss(nobs, B, deterministic=d)
        print dt.datetime.now() - now
        quantiles = np.percentile(kpss, list(percentiles))
        df = pd.DataFrame(quantiles, index=critical_values,columns=[d])
        df.to_hdf(hdf_filename, key=d, mode='a')
        quantiles = map(lambda x: '{0:0.5f}'.format(x), quantiles)

        sio.write("kpss_critical_values['" + d + "'] = (")
        count = 0
        for c, q in zip(critical_values_string, quantiles):
            sio.write('(' + c + ', ' + q + ')')
            count += 1
            if count % 3 == 0:
                sio.write(',\n                             ')
            else:
                sio.write(', ')
        sio.write(")")
        sio.write("\n")

    sio.seek(0)
    print sio.read()




