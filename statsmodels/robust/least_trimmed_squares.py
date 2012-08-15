# -*- coding: utf-8 -*-
"""

Created on Wed Jul 18 05:30:39 2012

Author: Josef Perktold
"""

import itertools

import numpy as np
from scipy.misc import comb

from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.discrete.discrete_model import Poisson
from robust_linear_model import RLM

class Holder(object):
    pass


def subsample(n, k, max_nrep=20):
    idx = np.ones(n, bool)
    idx[:(n-k)] = False
    for i in xrange(max_nrep):
        np.random.shuffle(idx)
        yield idx


def lts(endog, exog, k_trimmed=None, max_nstarts=5, max_nrefine=20, max_exact=100):
    nobs = endog.shape[0]
    nobs2, k_vars = exog.shape
    if k_trimmed is None:
        k_trimmed = nobs - int(np.trunc(nobs+k_vars)//2)
        #k_trimmed = nobs - (nobs - k_vars) + 1
    k_start = k_vars + 1
    k_accept = nobs - k_trimmed
    best = (np.inf, np.zeros(exog.shape[1]), np.nan * np.zeros(nobs))
    all_dict = {}
    n_est_calls = 0
    if comb(nobs, k_accept) <= max_exact:
        #index array
        iterator = itertools.combinations(range(nobs), k_accept)
    else:
        #boolean array
        #iterator = subsample(nobs, nobs - k_trimmed, max_nrep=max_nstarts)
        iterator = subsample(nobs, k_start, max_nrep=max_nstarts)
    for ii in iterator:
        if type(ii) is tuple:
            iin = np.zeros(nobs, bool)
            iin[list(ii)] = True
        else:
            iin = ii.copy()
        for ib in range(max_nrefine):
            res_t_ols = OLS(endog[iin], exog[iin]).fit()
            n_est_calls += 1
            #print np.nonzero(~iin)[0] + 1, res_t_ols.params, res_t_ols.ssr
            r = endog - res_t_ols.predict(exog)
            #ii2 = np.argsort(np.argsort(np.abs(r))) < k_accept
            idx3 = np.argsort(np.abs(r))[k_accept:]
            ii2 = np.ones(nobs, bool)
            ii2[idx3] = False
            if (ii2 == iin).all():
                if res_t_ols.ssr < best[0]:
                    #update best result so far

                    best = (res_t_ols.ssr, res_t_ols, ~ii2)
                break
            else:
                iin = ii2
                outl = tuple(np.nonzero(iin)[0])
                if outl in all_dict:
                    all_dict[outl] += 1
                    break
                else:
                    all_dict[outl] = 1
        else:
            print "maxiter 20 reached"

    best[1].all_dict = all_dict
    best[1].n_est_calls = n_est_calls
    return best


def scale_lts(ssr_trimmed, k_trimmed, nobs, k_vars, distr='norm'):
    '''calculate variance with correction for trimming

    Parameters
    ----------
    ssr_trimmed : float
        sum of squared residuals of observations included in trimmed estimation
    k_trimmed : int
        number of outlier candidates trimmed from the regression
    nobs : int
        total number of observations, included plus trimmed
    k_vars : int
        number of regressors (exog), used for small sample correction
    distr : string
        currently only 'norm' is supported

    '''

    if distr != 'norm':
        raise NotImplementedError('currently only normal distribution available')

    frac_trimmed = k_trimmed * 1.0 / nobs

    from scipy import stats
    crit = stats.norm.ppf(1 - frac_trimmed * 0.5)
    tau = 1 - frac_trimmed - 2 * crit * stats.norm.pdf(crit)

    correction_factor = (1 - frac_trimmed) / tau
    nobs_included = nobs - k_trimmed
    denom_bias_corrected =  (nobs_included - k_vars)

    return ssr_trimmed * correction_factor / denom_bias_corrected

#def scale_LTS():


class LTS(object):
    '''Least Trimmed Squares Estimation


    Parameters
    ----------
    endog : ndarray
        Y-variable, dependent variable
    exog : ndarray
        X-variable, independent, explanatory variables (regressors)
    est_model : model class
        default is OLS, needs to have fit and predict methods and attribute
        ssr
    fit_options : None or dict
        If fit_options are not None, then they will be used in the call to the
        fit method of the estimation model, ``est_model``


    TODO: variation: trim based on likelihood contribution, loglike_obs,
          and use llf instead of ssr
    '''


    def __init__(self, endog, exog, est_model=OLS):
        self.endog = endog
        self.exog = exog
        self.est_model = est_model
        self.nobs, self.k_vars = self.exog.shape
        #TODO: all_dict might not be useful anymore with new algorithm
        self.all_dict = {} #store tried outliers
        self.temp = Holder()
        self.temp.n_est_calls = 0
        self.temp.n_refine_steps = []

        self.target_attr = 'ssr'
        self.fit_options = {}

    def _refine_step(self, iin, k_accept):
        endog, exog = self.endog, self.exog
        nobs = self.nobs

        res_trimmed = self.est_model(endog[iin], exog[iin]).fit()
        self.temp.n_est_calls += 1
        #print np.nonzero(~iin)[0] + 1, res_t_ols.params, res_t_ols.ssr
        r = endog - res_trimmed.predict(exog)
        #ii2 = np.argsort(np.argsort(np.abs(r))) < k_accept
        #partial sort would be enough: need only the smallest k_outlier
        #values
        #TODO: another version: use resid_se and outlier test to determin
        #k_outliers
        idx3 = np.argsort(np.abs(r))[k_accept:] #idx of outliers

        ii2 = np.ones(nobs, bool)
        ii2[idx3] = False
        ssr_new = np.dot(r*r, ii2)
        return res_trimmed, ii2, ssr_new

    def refine(self, iin, k_accept, max_nrefine=2):
        '''
        concentration step
        '''

        endog, exog = self.endog, self.exog
        #nobs = self.nobs
        #all_dict = self.all_dict

        for ib in range(max_nrefine):
            #print tuple(np.nonzero(iin)[0])
            res_trimmed, ii2, ssr_new = self._refine_step(iin, k_accept)
            #print ib, tuple(np.nonzero(ii2)[0])
            if (ii2 == iin).all():
                converged = True
                break
            else:
                iin = ii2.copy()
                #for debugging
                outl = tuple(np.nonzero(iin)[0])
                all_dict = self.all_dict
                if outl in all_dict:
                    all_dict[outl] += 1
                else:
                    all_dict[outl] = 1
                 #remove stopping on already evaluated (seen before)
#                outl = tuple(np.nonzero(iin)[0])
#                if outl in all_dict:
#                    all_dict[outl] += 1
#                    break
#                else:
#                    all_dict[outl] = 1
        else:
            #max_nrefine reached
            converged = False

        self.temp.n_refine_steps.append(ib)
        return res_trimmed, ii2, ssr_new, converged


    def fit_random(self, k_trimmed, max_nstarts=10, k_start=None, n_keep=30,
                         max_nrefine_st1=1, max_nrefine_st2=100):
        '''find k_trimmed outliers with a 2-stage random search

        Parameters
        ----------
        k_trimmed : int
            number of observations to trim, i.e. number of outliers that are
            removed from the data for estimation
        max_nstarts : int
            number of random draws of outliers indices used in first stage
        k_start : int
            number of observations to include in the initial estimation in
            the first stage. Default is k_start = k_vars + 1 where k_vars is
            the number of explanatory (exog) variables.
        n_keep : int
            number of first stage results to use for the second stage, the
            concentration or refinement.
        max_nrefine_st1 : int
            maximum number of concentration or refinement steps in first stage.
            see Notes.
        max_nrefine_st2 : int
            maximum number of concentration or refinement steps in second stage.
            see Notes.

        Returns
        -------
        res_trimmed : results instance
            The best, lowest ssr, instance of the estimation results class
        ii2 : ndarray
            index of inliers, observations used in the best regression.

        Notes
        -----
        The heuristic for the random search follows approximately the algorithm
        of Rousseuw and van Driessen. In the first stage, a large number of
        random starts are evaluated. The best of those are used in a second
        stage until convergence.
        The first stage depends on three parameters, k_start, max_nstarts and
        max_nrefine_st1.
        The second stage depends only on n_keep, and an additional safeguard
        max_nrefine_st2.
        TODO: add max_nrefine to function arguments

        The number of estimations used is approximately
        max_nstarts * max_nrefine_st1 + n_keep * ave_refine_st2

        ave_refine_st2 is the average number of refinement steps needed until
        convergence and depends on the problem. In the test cases, it is very
        low for n_keep that are not large relative to max_nstarts (30%).

        TODO: renaming
        k_start -> nobs_start ?
        max_nstarts n_repl_starts ?
            max is not correct, it's always exactly max_nstarts

        The search itself, for fixed tuning parameter, does not increase much
        with nobs and k_trimmed. This does not include the increase in time for
        an estimation as nobs increases.

        n_keep : If n_keep is too low, for example 10 as in Rousseuw and Van Driessen,
            then the random search does not find the best solution for one of the
            test datasets (aircraft).

        k_start : some papers recommend k_start=k_vars for an exactly identified
            estimation problem, but that didn't work so well in my preliminary
            tests for OLS. (exactly identified cases might not work correctly
            for all models.)

        Currently, there is no check for a singular design matrix, exog. This
        could be the case if there are categorical variables.
        My guess is that it doesn't matter for the search process with OLS, but
        the final result could have a singular design matrix if the trimming
        is large enough. However this currently breaks with Poisson in subclass
        if the selected exog matrix is singular.

        '''
        #terminology in code: uses ssr, but this is nllf in LTLikelihood

        endog, exog = self.endog, self.exog
        nobs, k_vars = exog.shape  #instead of using attributes?

        if k_start is None:
            #TODO: check, this should be k_vars, exactly determined
            k_start = k_vars + 1
        k_accept = nobs - k_trimmed

        #stage 1
        best_stage1 = []
        ssr_keep = [np.inf] * n_keep
        #need sorted list that allows inserting and deletes worst
        #use: add if it is better than n_keep-worst (i.e. min_keep)

        iterator = subsample(nobs, k_start, max_nrep=max_nstarts)

        for ii in iterator:
            iin = ii.copy()   #TODO: do I still need a copy
            res = self.refine(iin, k_accept, max_nrefine=max_nrefine_st1)
            res_trimmed, ii2, ssr_new, converged = res
            #if res_trimmed.ssr < ssr_keep[n_keep-1]:
                #best_stage1.append((res_trimmed.ssr, ii2))
            if ssr_new < ssr_keep[n_keep-1]:
                best_stage1.append((ssr_new, ii2))
                #update minkeep, shouldn't grow longer than n_keep
                #we don't drop extra indices in best_stage1
                ssr_keep.append(ssr_new)
                ssr_keep.sort()  #inplace python sort
                del ssr_keep[n_keep:]   #remove extra

        #stage 2 : refine best_stage1 to convergence
        ssr_best = np.inf
        for (ssr, start_mask) in best_stage1:
            if ssr > ssr_keep[n_keep-1]: continue
            res = self.refine(start_mask, k_accept, max_nrefine=max_nrefine_st2)
            res_trimmed, ii2, ssr_new, converged = res
            if not converged:
                #warning ?
                print "refine step did not converge, max_nrefine limit reached"

            ssr_current = getattr(res_trimmed, self.target_attr)  #res_trimmed.ssr
            if ssr_current < ssr_best:
                ssr_best = ssr_current
                res_best = (res_trimmed, ii2)

        self.temp.best_stage1 = best_stage1
        self.temp.ssr_keep = ssr_keep

        return res_best

    def fit_exact(self, k_trimmed=None):
        endog, exog = self.endog, self.exog
        nobs = self.nobs
        k_accept = nobs - k_trimmed
        iterator = itertools.combinations(range(nobs), k_accept)

        value_best = np.inf
        for keep_idx in iterator:
            #indexing with list
            keep_idx = list(keep_idx)
            endog_ = endog.take(keep_idx)
            exog_ = exog.take(keep_idx, axis=0)
            res_trimmed = self.est_model(endog_, exog_).fit(**self.fit_options)
            self.temp.n_est_calls += 1

            value_current = getattr(res_trimmed, self.target_attr)
            if value_current < value_best:
                value_best = value_current
                res_best = (res_trimmed, keep_idx)

        ii2 =  np.zeros(nobs, bool)
        ii2[list(res_best[1])] = True
        return (res_best[0], ii2)

    def fit(self, k_trimmed=None, max_exact=100, random_search_options=None):
        '''find k_trimmed outliers with a 2-stage random search

        Parameters
        ----------
        k_trimmed : int
            number of observations to trim, i.e. number of outliers that are
            removed from the data for estimation
        max_exact : int
            If the number of estimations for the exact case, i.e. full
            enumeration of all possible outlier constellations, is below
            max_exact, then the exact method is used. If it is larger, then
            then fit_random is used.
            NotImplemented yet
        random_search_options : dict
            options that are used in fit_random, see fit_random for details

        Returns
        -------
        res_trimmed : results instance
            The best, lowest ssr, instance of the estimation results class.
            Warning: this is just the OLS Results instance with the trimmed
            set of observations and does not take the trimming into account.
            TODO: adjust results, especially scale
        ii2 : ndarray
            index of inliers, observations used in the best regression.

        Notes
        -----
        see fit_random

        '''
        nobs, k_vars = self.nobs, self.k_vars

        if k_trimmed is None:
            k_trimmed = nobs - int(np.trunc(nobs + k_vars)//2)

        self.k_accept = k_accept = nobs - k_trimmed

        if comb(nobs, k_accept) <= max_exact:
            #index array
            self.options_fit_used = ('exact', {})
            return self.fit_exact(k_trimmed)
            #iterator = itertools.combinations(range(nobs), k_accept)
        else:
            #boolean array
            options = dict(max_nstarts=10, k_start=None, n_keep=10)
            if not random_search_options is None:
                options.update(random_search_options)
            self.options_fit_used = ('exact', options)
            return self.fit_random(k_trimmed, **options)


from statsmodels.discrete.discrete_model import DiscreteResults
#DiscreteResults.nllf = lambda : - DiscreteResults.llf

class MaximumTrimmedLikelihood(LTS):

    def __init__(self, endog, exog, est_model=Poisson, fit_options=None):
        super(LTLikelihood, self).__init__(endog, exog, est_model=est_model)
        #patching model doesn't help
        #self.est_model.nllf =
        self.target_attr = 'nllf'
        fit_options_ = dict(disp=False)
        if not fit_options is None:
            fit_options_.update(fit_options)
        self.fit_options = fit_options_

    def _refine_step(self, iin, k_accept):
        endog, exog = self.endog, self.exog
        nobs = self.nobs

        res_trimmed = self.est_model(endog[iin], exog[iin]).fit(
                                                            **self.fit_options)
        self.temp.n_est_calls += 1
        #print np.nonzero(~iin)[0] + 1, res_t_ols.params, res_t_ols.ssr
        #r = endog - res_trimmed.predict(exog)

        #TODO: more monkey, need loglike for all observations
        res_trimmed.model.endog = endog
        res_trimmed.model.exog = exog

        r = res_trimmed.model.loglikeobs(res_trimmed.params)
        #ii2 = np.argsort(np.argsort(np.abs(r))) < k_accept
        #partial sort would be enough: need only the smallest k_outlier
        #values
        #TODO: another version: use resid_se and outlier test to determin
        #k_outliers
        idx3 = np.argsort(np.abs(r))[k_accept:] #idx of outliers

        ii2 = np.ones(nobs, bool)
        ii2[idx3] = False
        ssr_new = np.dot(r*r, ii2)
        return res_trimmed, ii2, ssr_new

class EfficientLTS(object):

    '''efficient least trimmed squares

    this follows Doornik 2011
    experimental, to see if it works
    converted to class to store intermediate results for debugging

    Parameters
    ----------
    endog : ndarray, 1d
        independent variable
    exog : ndarray, (nobs, k_vars)
        explanatory variables
    breakdown : float in (0,1)
        fraction of outliers in initial LTS
    efficiency : float in (0, 1)
        efficiency parameter for final trimming parameters
    maxiter : int
        maximum number of iterations to perform. Iteration will stop before if
        outliers don't change across two consecutive iterations.


    Notes
    -----
    section 3.1 in Doornik

    Notation:

    breakdown = alpha1
    efficiency = 1 - alpha2

    '''
    def __init__(self, endog, exog):
        self.endog = endog
        self.exog = exog

    def fit(self, breakdown, efficiency, random_search_options=None, maxiter=10):
        '''
        #check breakdown and efficiency names are not completely right
        #actual efficiency is lower that the efficiency parameter

        Parameters
        ----------
        breakdown : float in (0,1)
            fraction of outliers in initial LTS
        efficiency : float in (0, 1)
            efficiency parameter for final trimming parameters
            The actual assymptotic efficiency is tau (attached to model
            instance).
        maxiter : int
            maximum number of iterations to perform. Iteration will stop before if
            outliers don't change across two consecutive iterations.
        random_search_options : None or dict
            option for the initial LTS random search, see LTS.fit_random for
            details

        Returns
        -------
        res_ols : OLSResults instance
            OLS results from last estimation with trimmed endog and exog,
            not corrected for ELTS, i.e. no scale adjustment yet
            TODO: add scale adjustment and test
        keep_mask : ndarray of boolean
            Mask of observation in the original sample that are included in the
            final regression.

        Notes
        -----
        The asymptotic efficiency based on the efficiency parameter, 1 - alpha2,
        is given by the attribute tau, see Doornik around equation 4, if the
        underlying distribution of the errors is assumed to be normal. Normal
        distribution is the only case that is implemented.

        Asymptotic distribution of parameters of LTS is normal using the
        adjusted scale. For the ELTS extension this is conjectured
        (Doornik equation (7)).

        '''

        endog = self.endog
        exog = self.exog
        alpha2 = 1 - efficiency

        nobs, k_vars = exog.shape
        k_trimmed = int(nobs * breakdown)   #truncate, round down
        res_lts, keep_mask = LTS(endog, exog).fit(k_trimmed=k_trimmed,
                                    random_search_options=random_search_options)
        fittedvalues_all = res_lts.predict(exog)
        resid = endog - fittedvalues_all
        scale_adj = scale_lts(res_lts.ssr, (~keep_mask).sum(), nobs, k_vars,
                          distr='norm')

        from scipy import stats
        crit = stats.norm.ppf(1 - alpha2/2.)
        tau = efficiency - 2 * crit * stats.norm.pdf(crit)  #kept constant
        correction_factor = (efficiency) / tau

        self.crit = crit
        self.tau = tau
        self.correction_factor = correction_factor

        keep_mask_old = np.zeros(nobs)   #check structure of loop
        for it in range(maxiter):
            resid_scaled = resid / np.sqrt(scale_adj)
            #outl = np.nonzero[(np.abs(resid_scaled) > crit)] #optional
            keep_mask = (np.abs(resid_scaled) < crit)
            if not np.any(keep_mask != keep_mask_old):  #converged
                break

            keep_mask_old = keep_mask
            #use WLS with 0-1 weight instead of trimmed OLS
            #res_wls = WLS(endog, exog, weights=weights).fit()
            res_ols = OLS(endog[keep_mask], exog[keep_mask]).fit()
            scale_adj = correction_factor * res_ols.scale
            #is there ddof correction (nobs-k_vars) in res_ols.scale? I need it
            fittedvalues_all = res_ols.predict(exog)
            resid = endog - fittedvalues_all

        self.iterations = it
        self.scale_adj = scale_adj
        return res_ols, keep_mask

class LTSRLM(RLM):
    '''Robust estimation, MM-estimator with LTS as starting value

    '''


    def fit(self, breakdown=0.5, random_search_options=None, rlm_args=None):
        if rlm_args is None:
            rlm_args = {}

        endog = self.endog
        exog = self.exog

        nobs, k_vars = exog.shape
        k_trimmed = int(nobs * breakdown)   #truncate, round down
        res_lts, keep_mask = LTS(endog, exog).fit(k_trimmed=k_trimmed,
                                    random_search_options=random_search_options)
        fittedvalues_all = res_lts.predict(exog)
        resid = endog - fittedvalues_all
        scale_adj = scale_lts(res_lts.ssr, (~keep_mask).sum(), nobs, k_vars,
                          distr='norm')

        # fixed scale during M-estimation

        init = dict(scale=scale_adj)
        res_rlm = super(LTSRLM, self).fit(weights=keep_mask.astype(float),
                                          update_scale=False,
                                          init=init, #for scale
                                          **rlm_args)

        res_rlm.results_lts = res_lts
        res_rlm.results_lts_keep_mask = keep_mask
        return res_rlm
