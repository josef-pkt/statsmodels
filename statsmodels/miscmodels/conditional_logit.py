# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:54:41 2015

Author: Josef Perktold
License: BSD-3
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:46:41 2015

Author: Josef Perktold
License: BSD-3
"""

import numpy as np
from statsmodels.base.model import GenericLikelihoodModel



def prob_total(ni, ki, expxb):
    # a lower memory version - keep only 2 f vectors columns of full f
    f2_orig = np.zeros(ni+1)

    f1 = np.ones(ni+1)
    for k in range(1, ki+1):

        f2 = f2_orig.copy()
        for t in range(k, ni+1):
            f2[t] = f2[t-1] + f1[t-1] * expxb[t-1]
        print(f2)
        f1 = f2

    return f2

def loglike_one_groups(endog, xb, group_idx):
    """loglike without standard interface
    """
    # use boolean for indexing with mask
    endog = np.asarray(endog, np.bool_)
    expxb_groups = np.add.reduceat(np.exp(xb), group_idx)
    return xb[endog] - np.log(expxb_groups)

def score_one_groups(params, endog, exog, group_idx):
    """loglike without standard interface
    """
    # use boolean for indexing with mask
    endog = np.asarray(endog, np.bool_)
    xb = exog.dot(params)
    expxb_groups = np.add.reduceat(np.exp(xb), group_idx)
    expxb_groups_dp = np.add.reduceat(np.exp(xb)[:,None]*exog, group_idx,
                                      axis=0)
    return exog[endog] - (expxb_groups_dp / expxb_groups[:,None])

# ignore for now
def loglike_one_obs(endog, xb, group_idx, groups):
    """this doesn't make sense
    check: we are subtracting log sum_i expxb for each observation

    can we define an contribution of an observation?
    - only by faking it for numerical convenience? which is just inefficient
    - contribution of not selected observations is zero ?
    how do we calculate sandwich cov_params for cluster robust?
    Should work automatically from sum of score over within cluster observations.
    """
    # use boolean for indexing with mask
    endog = np.asarray(endog, np.bool_)
    expxb_groups = np.add.reduceat(np.exp(xb), group_idx)
    ll = - np.log(expxb_groups[groups])
    ll[endog] += xb[endog]
    return ll #endog * xb - expxb_groups[groups]


class ConditionalLogitOne(GenericLikelihoodModel):

    def __init__(self, endog, exog, groups, **kwds):
        super(ConditionalLogitOne, self).__init__(endog, exog, **kwds)

        self.endog = np.asarray(endog, bool)
        self.groups = np.asarray(groups)
        self.group_idx = np.concatenate(([0], np.where(groups[1:] != groups[:-1])[0] + 1))


    def loglike(self, params):

        xb = self.exog.dot(params)
        return loglike_one_groups(self.endog, xb, self.group_idx).sum()

    def loglike_obs(self, params):

        xb = self.exog.dot(params)
        return loglike_one_groups(self.endog, xb, self.group_idx)


    def predict(self, params, exog=None, conditional=True):
        # only partially checked
        # Todo : need group_idx corresponding to user provided exog
        # currently works only for sample exog

        # Todo: What we want to do is to predict on individual and not one
        # observation. So either require groups indicator
        # or assume all exog are from the same group/individual/strata.
        if exog is None:
            exog = self.exog
        xb = exog.dot(params)

        if conditional:
            expxb = np.exp(xb)
            expxb_groups = np.add.reduceat(expxb, self.group_idx)
            prob = expxb / expxb_groups[self.groups]
        else:
            prob = 1 / (1 + np.exp(-xb))

        return prob

    def deriv_predict_dx(self, params, exog=None, conditional=True):
        # only partially checked
        # see comments in predict about treatment of exog
        # the derivative assumes that the exog change for one observation (choice),
        # not for one individual.
        # Important if conditional is true, which keeps other observations of
        # the same individual unchanged.
        if exog is None:
            exog = self.exog
        xb = exog.dot(params)

        prob = self.predict(params, exog=exog, conditional=conditional)
        return (prob * (1 - prob))[:, None] * params

        if conditional:
            # this part doesn't work yet
            expxb = np.exp(xb)
            expxb_groups = np.add.reduceat(np.exp(xb), self.group_idx)
            expxb_groups_dx = np.add.reduceat(np.exp(xb)[:,None]*params,
                                              self.group_idx, axis=0)
            deriv = (expxb_groups_dx / expxb_groups[:,None])
            deriv = expxb[:,None] * params / expxb_groups[self.groups]
        else:
            deriv = (np.exp(-xb) / (1 + np.exp(-xb))**2)[:,None] * params

        return deriv


    def predict_one_groups(self, params, exog):
        """predicted conditional choice probabilities for one group

        this is a temporary method and will be merged into predict.

        This is conditional that the group has to select exactly one choice.
        """
        expxb = np.exp(exog.dot(params))

        return expxb / expxb.sum()


    def fit(self, **kwds):
        # tODO: fix list of kwds to explicit
        # only used to change default method to bfgs
        method = kwds.pop('method', 'bfgs')
        return super(ConditionalLogitOne, self).fit(method=method, **kwds)
