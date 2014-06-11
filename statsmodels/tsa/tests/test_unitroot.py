# -*- coding: utf-8 -*-
"""Test for autolag of adfuller, unitroot_adf

Created on Wed May 30 21:39:46 2012
Author: Josef Perktold
"""
from statsmodels.compat.python import iteritems
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import statsmodels.tsa.stattools as tsast
from statsmodels.datasets import macrodata

def test_adf_autolag():
    #see issue #246
    #this is mostly a unit test
    d2 = macrodata.load().data

    for k_trend, tr in enumerate(['nc', 'c', 'ct', 'ctt']):
        #[None:'nc', 0:'c', 1:'ct', 2:'ctt']
        x = np.log(d2['realgdp'])
        xd = np.diff(x)

        #check exog
        adf3 = adfuller(x, maxlag=None, autolag='aic',
                              regression=tr, store=True, regresults=True)
        st2 = adf3[-1]

        assert_equal(len(st2.autolag_results), 15 + 1)  #+1 for lagged level
        for l, res in sorted(iteritems(st2.autolag_results))[:5]:
            lag = l-k_trend
            #assert correct design matrices in _autolag
            assert_equal(res.model.exog[-10:,k_trend], x[-11:-1])
            assert_equal(res.model.exog[-1,k_trend+1:], xd[-lag:-1][::-1])
            #min-ic lag of dfgls in Stata is also 2, or 9 for maic with notrend
            assert_equal(st2.usedlag, 2)

        #same result with lag fixed at usedlag of autolag
        adf2 = adfuller(x, maxlag=2, autolag=None, regression=tr)
        assert_almost_equal(adf3[:2], adf2[:2], decimal=12)


    tr = 'c'
    #check maxlag with autolag
    adf3 = adfuller(x, maxlag=5, autolag='aic',
                          regression=tr, store=True, regresults=True)
    assert_equal(len(adf3[-1].autolag_results), 5 + 1)
    adf3 = adfuller(x, maxlag=0, autolag='aic',
                          regression=tr, store=True, regresults=True)
    assert_equal(len(adf3[-1].autolag_results), 0 + 1)


class CheckADF(object):
    """
    Test Augmented Dickey-Fuller

    Test values taken from Stata.
    """
    levels = ['1%', '5%', '10%']
    data = macrodata.load()
    x = data.data['realgdp']
    y = data.data['infl']

    def test_teststat(self):
        assert_almost_equal(self.res1[0], self.teststat, DECIMAL_5)

    def test_pvalue(self):
        assert_almost_equal(self.res1[1], self.pvalue, DECIMAL_5)

    def test_critvalues(self):
        critvalues = [self.res1[4][lev] for lev in self.levels]
        assert_almost_equal(critvalues, self.critvalues, DECIMAL_2)

class TestADFConstant(CheckADF):
    """
    Dickey-Fuller test for unit root
    """
    def __init__(self):
        self.res1 = adfuller(self.x, regression="c", autolag=None,
                maxlag=4)
        self.teststat = .97505319
        self.pvalue = .99399563
        self.critvalues = [-3.476, -2.883, -2.573]

class TestADFConstantTrend(CheckADF):
    """
    """
    def __init__(self):
        self.res1 = adfuller(self.x, regression="ct", autolag=None,
                maxlag=4)
        self.teststat = -1.8566374
        self.pvalue = .67682968
        self.critvalues = [-4.007, -3.437, -3.137]

#class TestADFConstantTrendSquared(CheckADF):
#    """
#    """
#    pass
#TODO: get test values from R?

class TestADFNoConstant(CheckADF):
    """
    """
    def __init__(self):
        self.res1 = adfuller(self.x, regression="nc", autolag=None,
                maxlag=4)
        self.teststat = 3.5227498
        self.pvalue = .99999 # Stata does not return a p-value for noconstant.
                        # Tau^max in MacKinnon (1994) is missing, so it is
                        # assumed that its right-tail is well-behaved
        self.critvalues = [-2.587, -1.950, -1.617]

# No Unit Root

class TestADFConstant2(CheckADF):
    def __init__(self):
        self.res1 = adfuller(self.y, regression="c", autolag=None,
                maxlag=1)
        self.teststat = -4.3346988
        self.pvalue = .00038661
        self.critvalues = [-3.476, -2.883, -2.573]

class TestADFConstantTrend2(CheckADF):
    def __init__(self):
        self.res1 = adfuller(self.y, regression="ct", autolag=None,
                maxlag=1)
        self.teststat = -4.425093
        self.pvalue = .00199633
        self.critvalues = [-4.006, -3.437, -3.137]

class TestADFNoConstant2(CheckADF):
    def __init__(self):
        self.res1 = adfuller(self.y, regression="nc", autolag=None,
                maxlag=1)
        self.teststat = -2.4511596
        self.pvalue = 0.013747 # Stata does not return a p-value for noconstant
                               # this value is just taken from our results
        self.critvalues = [-2.587,-1.950,-1.617]

