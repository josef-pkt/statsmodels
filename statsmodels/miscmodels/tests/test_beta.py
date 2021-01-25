from __future__ import print_function
import io
import os.path as op
import sys
from statsmodels.api import families
links = families.links
import pandas as pd
import patsy
from statsmodels.miscmodels.betareg import Beta
import numpy as np

HERE = op.abspath(op.dirname(__file__))

# betareg(I(food/income) ~ income + persons, data = FoodExpenditure)
_income_estimates_mean = u"""\
varname        Estimate  StdError   zvalue     Pr(>|z|)
(Intercept) -0.62254806 0.223853539 -2.781051 5.418326e-03
income      -0.01229884 0.003035585 -4.051556 5.087819e-05
persons      0.11846210 0.035340667  3.352005 8.022853e-04"""

_income_estimates_precision = u"""\
varname  Estimate StdError  zvalue     Pr(>|z|)
(phi) 35.60975   8.079598 4.407366 1.046351e-05
"""

_methylation_estimates_mean = u"""\
varname      Estimate StdError zvalue Pr(>|z|)
(Intercept)  1.44224    0.03401  42.404   2e-16
genderM      0.06986    0.04359   1.603    0.109
CpGCpG_1     0.60735    0.04834  12.563   2e-16
CpGCpG_2     0.97355    0.05311  18.331   2e-16"""

_methylation_estimates_precision = u"""\
varname Estimate StdError zvalue Pr(>|z|)
(Intercept)  8.22829    1.79098   4.594 4.34e-06
age         -0.03471    0.03276  -1.059    0.289"""


expected_income_mean = pd.read_table(io.StringIO(_income_estimates_mean), sep="\s+")
expected_income_precision = pd.read_table(io.StringIO(_income_estimates_precision), sep="\s+")

expected_methylation_mean = pd.read_table(io.StringIO(_methylation_estimates_mean), sep="\s+")
expected_methylation_precision = pd.read_table(io.StringIO(_methylation_estimates_precision), sep="\s+")

income = pd.read_csv(op.join(HERE, 'foodexpenditure.csv'))
methylation = pd.read_csv(op.join(HERE, 'methylation-test.csv'))

def check_same(a, b, eps, name):
    assert np.allclose(a, b, rtol=0.01, atol=eps), \
            ("different from expected", name, list(a), list(b))

def assert_close(a, b, eps):
    assert np.allclose(a, b, rtol=0.01, atol=eps), (list(a), list(b))

class TestBeta(object):

    @classmethod
    def setupClass(self):
        model = "I(food/income) ~ income + persons"
        self.income_fit =Beta.from_formula(model, income).fit()

        model = self.model = "methylation ~ gender + CpG"
        Z = self.Z = patsy.dmatrix("~ age", methylation)
        self.meth_fit = Beta.from_formula(model, methylation, exog_precision=Z,
                                          link_precision=links.identity()).fit()

    def test_income_coefficients(self):
        rslt = self.income_fit
        assert_close(rslt.params[:-1], expected_income_mean['Estimate'], 1e-3)
        assert_close(rslt.tvalues[:-1], expected_income_mean['zvalue'], 0.1)
        assert_close(rslt.pvalues[:-1], expected_income_mean['Pr(>|z|)'], 1e-3)


    def test_income_precision(self):

        rslt = self.income_fit
        # note that we have to exp the phi results for now.
        assert_close(np.exp(rslt.params[-1:]), expected_income_precision['Estimate'], 1e-3)
        #yield check_same, rslt.tvalues[-1:], expected_income_precision['zvalue'], 0.1, "z-score"
        assert_close(rslt.pvalues[-1:], expected_income_precision['Pr(>|z|)'], 1e-3)


    def test_methylation_coefficients(self):
        rslt = self.meth_fit
        assert_close(rslt.params[:-2], expected_methylation_mean['Estimate'], 1e-2)
        assert_close(rslt.tvalues[:-2], expected_methylation_mean['zvalue'], 0.1)
        assert_close(rslt.pvalues[:-2], expected_methylation_mean['Pr(>|z|)'], 1e-2)

    def test_methylation_precision(self):
        rslt = self.meth_fit

        #yield check_same, links.logit()(rslt.params[-2:]), expected_methylation_precision['Estimate'], 1e-3, "estimate"
        #yield check_same, rslt.tvalues[-2:], expected_methylation_precision['zvalue'], 0.1, "z-score"

    def test_precision_formula(self):
        m = Beta.from_formula(self.model, methylation, exog_precision_formula='~ age',
                                          link_precision=links.identity())
        rslt = m.fit()
        assert_close(rslt.params, self.meth_fit.params, 1e-10)


    def test_scores(self):
        model, Z = self.model, self.Z
        for link in (links.identity(), links.log()):
            mod2 = Beta.from_formula(model, methylation, exog_precision=Z,
                                     link_precision=link)
            rslt_m = mod2.fit()

            analytical = rslt_m.model.score(rslt_m.params)
            numerical = rslt_m.model.score_check(rslt_m.params)
            assert_close(analytical[:3], numerical[:3], 1e-2)
            assert_close(link.inverse(analytical[3:]), link.inverse(numerical[3:]), 1e-2)
