# The rest of the points are coupon bonds. We assume that the YTM given for the bonds are all par rates.
# So we have bonds with coupon rate same as the YTM.

import QuantLib2 as ql
import Quantlib_Helpers

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np
from enum import Enum

_nullSolution = Quantlib_Helpers.ql_array([-5.0,  5.0, 3.1,  10.0, 1.0, 1.0]) # Decent guess for average fit

class FitMethod(Enum):
    NELDER_MEAD = 0
    DE = 1
    ANNEALING = 2



# Wrapper for parallel computation
# Passing Pandas objects around in a multithreaded context ruins any speed improvements.
# Also Quantlib objects can't be serialized so they can't be passed through Dask.
def fit_securities_quantlib(bond_maturities, itype, bond_rates, bond_prices, cusips, bond_tdatdt, obs_date):
    """
    :param bond_maturities:
    :param itype:
    :param bond_rates:
    :param bond_prices:
    :param cusips:
    :param bond_tdatdt:
    :param obs_date:
    :return: [self.obs_date, self.cusips, self.cleanPrices, self.bond_prices_out,
                self.yieldcurve_output, self.bond_yields, self.fit_err_std]
    """
    fit = FitSecuritiesQuantlib(bond_maturities, itype, bond_rates, bond_prices, cusips, bond_tdatdt, obs_date)
    return fit.fit()


def fit_securities_quantlib_param(bond_maturities, itype, bond_rates, bond_prices, cusips, bond_tdatdt, obs_date, previous_solution = _nullSolution):
    fit = FitSecuritiesQuantlib(bond_maturities, itype, bond_rates, bond_prices, cusips, bond_tdatdt, obs_date)
    res = fit.fit(previous_solution)
    paramSolution = res.yieldcurve.fitResults().solution()
    return res, paramSolution



def _make_bond_orig(opt, price, rate, issue_date, maturity_date):
    coupon = [rate / 100.0]
    if coupon is 0:
        coupon_frequency = ql.Period(ql.Once)
    else:
        coupon_frequency = opt.coupon_frequency
    schedule = ql.Schedule(issue_date,
                           maturity_date,
                           coupon_frequency,
                           opt.calendar,
                           opt.business_convention,
                           opt.business_convention,
                           ql.DateGeneration.Backward,
                           opt.end_of_month)
    bond_helper = ql.FixedRateBondHelper(ql.QuoteHandle(ql.SimpleQuote(price)),
                                         opt.settlement_days,
                                         opt.face_amount,
                                         schedule,
                                         coupon,
                                         opt.day_counter,
                                         opt.business_convention)
    fixed_rate_bond = ql.FixedRateBond(opt.settlement_days,
                                       opt.face_amount,
                                       schedule,
                                       coupon,
                                       opt.day_counter,
                                       opt.business_convention,
                                       100.0)
    return bond_helper, fixed_rate_bond

# I really need source control....
def _make_bond(opt, price, coupon, issue_date, maturity_date):
    coupon = [coupon / 100.0]
    if coupon is 0:
        bond = ql.ZeroCouponBond( opt.settlement_days,
                                  opt.day_calendar,
                                  maturity_date,
                                  opt.business_convention,
                                  100.0 )
    else:
        schedule = ql.Schedule(issue_date,
                               maturity_date,
                               ql.Period(opt.coupon_frequency),
                               opt.calendar,
                               opt.business_convention,
                               opt.business_convention,
                               ql.DateGeneration.Backward,
                               opt.end_of_month)
        bond = ql.FixedRateBond(opt.settlement_days,
                                opt.face_amount,
                                schedule,
                                coupon,
                                opt.day_counter,
                                opt.business_convention,
                                100.0)
    bond_helper = ql.BondHelper(ql.QuoteHandle(ql.SimpleQuote(price)), bond)
    return bond_helper, bond



def _yield_for_bond(b: ql.Bond, p, r, settleDate, opt: Quantlib_Helpers.QuantlibFitOptions):
    # Bill and bond yields are different!
    if r == 0:
        return b.bondYield(p, opt.day_counter, ql.Simple, ql.Once, settleDate)
    else:
        return b.bondYield(p, opt.day_counter, opt.compounding, opt.coupon_frequency, settleDate)




class FitSecuritiesQuantlib:
    import Quantlib_Helpers
    opt = Quantlib_Helpers.QuantlibFitOptions()

    # Public members
    obs_date = None
    obs_date_ql = None
    cusips = None
    cleanPrices = None  # Estimated
    bond_prices = None
    bond_prices_out = None
    yieldcurve_output = None  # Three columns: maturity_times, zc_discounts, zero_curve
    bond_yields = None  # Two columns: [Observed Estimated]
    fit_err_std = None
    price_err_rmse = None
    itype = None
    check_problems = False # Turn this on for GovPX
    nss_params_out = None
    fit_method:FitMethod = FitMethod.NELDER_MEAD
    regularize_prior = None # Penalize params?

    fit_err = None
    large_err = None
    bond_maturities = None
    bond_tdatdt = None
    bond_yields_observed = None  # Duplicated in bond_yields

    # Quantlib stuff. Maybe do not save as class members?
    # Need to declare these before we construct bonds
    discountingTermStructure = None
    bondPricingEngine = None
    yieldcurve = None
    bonds = None
    bond_helpers = None
    end_criterion:ql.EndCriteria = None
    min_value = None
    equal_weight = False

    # We will hold onto these
    bond_coupon_rates = None
    bad_cusips = None
    bad_bond_info = None

    # Plotting bonds is tricky
    bond_plot_mats = None
    bond_plot_yields = None

    def __init__(self, bond_maturities, itype, bond_coupon_rates, bond_prices, cusips, bond_tdatdt, obs_date):
        self.obs_date = obs_date
        self.obs_date_ql = Quantlib_Helpers.timestamp_to_qldate(obs_date)
        drop = ((bond_maturities < 28) | (bond_maturities > 3700) | ((bond_maturities < 366) & (itype != 4)) | (itype > 4))
        self.bond_maturities = bond_maturities[~drop]
        self.bond_coupon_rates = bond_coupon_rates[~drop]
        self.bond_prices = bond_prices[~drop]
        self.cusips = cusips[~drop]
        self.bond_tdatdt = bond_tdatdt[~drop]
        self.itype = itype[~drop]
        self.bad_cusips = []
        ql.Settings.instance().evaluationDate = self.obs_date_ql  # todo: why global scope?
        self.bond_plot_mats = np.array([self.opt.day_counter.yearFraction(self.obs_date_ql, self.obs_date_ql + m)
                                        for m in self.bond_maturities])
        self._init_bonds()

    def _init_bonds(self):
        # Assemble bond information to construct Quantlib BondHelper and FixedRateBond objects
        self.discountingTermStructure = ql.RelinkableYieldTermStructureHandle()
        self.bondPricingEngine = ql.DiscountingBondEngine(self.discountingTermStructure)
        self.bonds = []
        self.bond_helpers = []
        self.bond_yields_observed = []
        self.bond_plot_yields = []
        for r, m, p, cusip, tdatdt in zip(self.bond_coupon_rates, self.bond_maturities,
                                          self.bond_prices, self.cusips, self.bond_tdatdt):
            # maturity_date = self.obs_date_ql + ql.Period(m, ql.Days)
            issue_date = Quantlib_Helpers.timestamp_to_qldate(tdatdt)
            maturity_date = self.obs_date_ql + m
            bond_helper, fixed_rate_bond = _make_bond(self.opt, p, r, issue_date, maturity_date)
            self.bond_helpers.append(bond_helper)
            fixed_rate_bond.setPricingEngine(self.bondPricingEngine)
            self.bonds.append(fixed_rate_bond)
            self.bond_yields_observed.append(_yield_for_bond(fixed_rate_bond, p, r, self.obs_date_ql, self.opt))
            if r is 0:
                self.bond_plot_yields.append(100.0 / p - 1.0)
            else:
                self.bond_plot_yields.append(self.bond_yields_observed[-1])
        self.bond_yields_observed = np.array(self.bond_yields_observed)
        self.bond_plot_yields = np.array(self.bond_plot_yields)

    def fit(self, previous_solution=None, bounds = None):
        """Calculates fit."""
        ## Idea: identify which bonds give us large fitting errors?
        self._fit_yields(previous_solution, bounds)
        self._fit_results()
        if sum(self.large_err > 0) and self.check_problems:
            # Delete bonds that don't fit and start over
            self.bad_cusips = self.cusips[self.large_err]
            self.bad_bond_info = {'maturities': self.bond_maturities[self.large_err],
                             'prices': self.bond_prices[self.large_err],
                             'bondYields': self.bond_yields_observed[self.large_err],
                             'plot_mats': self.bond_plot_mats[self.large_err]}
            # print("Large pricing error for " + str(self.bad_cusips) + " on " + str(pd.to_datetime(self.obs_date)))
            self._drop_bad(self.large_err)
            self._init_bonds()
            self._fit_yields(previous_solution, bounds)
            self._fit_results()
        return self.get_output()

    def fit_quickly(self, previous_solution = None, bounds=None):
        self._fit_yields(previous_solution, bounds)

        return {'min_value': self.yieldcurve.fitResults().minimumCostValue(),
                'end_criterion': self.yieldcurve.fitResults().endCriterion(),
                'nss_params': self._params()}

    def fit_quickly_finalize(self):
        self._fit_results()
        return self.get_output()


    def _params(self):
        # TODO: make sure we did an estimation already
        return list(self.yieldcurve.fitResults().solution())

    def assign_nss_params(self, params):
        self.assign_nss_params_quickly(params)
        self._fit_results()
        return self.get_output()

    def assign_nss_params_quickly(self, params):
        NSS = ql.SvenssonFitting()
        tolerance = 1.0e-12
        self.yieldcurve = ql.FittedBondDiscountCurve(self.obs_date_ql, self.bond_helpers,
                                                     self.opt.day_counter, NSS, tolerance, 0, params)
        self.yieldcurve.enableExtrapolation()
        self.discountingTermStructure.linkTo(self.yieldcurve)
        return self.yieldcurve.fitResults().minimumCostValue()

    def _drop_bad(self, ind):
        self.bond_maturities = self.bond_maturities[~ind]
        self.bond_coupon_rates = self.bond_coupon_rates[~ind]
        self.bond_prices = self.bond_prices[~ind]
        self.cusips = self.cusips[~ind]
        self.bond_tdatdt = self.bond_tdatdt[~ind]
        self.bond_plot_mats = self.bond_plot_mats[~ind]
        self.itype = self.itype[~ind]
        self.bonds = None
        self.bond_helpers = None
        self.bond_yields_observed = None

    def get_output(self):
        return {"obs_date": self.obs_date, "cusips": self.cusips, "bond_prices": self.bond_prices_out,
                "bond_yields": self.bond_yields, "yieldcurve": self.yieldcurve_output,
                "fit_err": self.fit_err_std, "price_err": self.price_err_rmse, "bad_cusips": self.bad_cusips,
                "nss_params": self.nss_params_out, "end_criterion": self.end_criterion,
                "min_value": self.min_value}


    def _fit_results(self):
        maturity_dates = [self.obs_date_ql + ql.Period(m, ql.Months) for m in range(1, 121)]
        maturity_times = [self.opt.day_counter.yearFraction(self.obs_date_ql, m) for m in maturity_dates]
        yieldcurve_discount = np.array([self.yieldcurve.discount(m) for m in maturity_times], dtype=float)
        yieldcurve_zeros = np.array([self.yieldcurve.zeroRate(m, ql.Compounded).rate() for m in maturity_times], dtype=float)
        self.yieldcurve_output = np.stack([maturity_times, yieldcurve_discount, yieldcurve_zeros], axis=1)
        self.cleanPrices = np.array([b.cleanPrice() for b in self.bonds], dtype=float) # These are estimated
        fit_yields = []
        for b, r, p_est, y_obs in zip(self.bonds, self.bond_coupon_rates, self.cleanPrices, self.bond_yields_observed):
            fit_yields.append([y_obs, _yield_for_bond(b, p_est, r, self.obs_date_ql, self.opt)])
        self.bond_yields = np.array(fit_yields)
        self.fit_err = np.array(self.bond_yields[:, 1] - self.bond_yields[:, 0])
        err_std = np.std(self.fit_err)
        self.large_err = np.abs(self.fit_err) > (4 * err_std)
        self.fit_err_std = np.sqrt(np.mean(np.square(self.fit_err[(~self.large_err) & (np.array(self.bond_maturities) > 1.0)])))
        bond_price_err = self.bond_prices - self.cleanPrices
        self.price_err_rmse = np.sqrt(np.mean(np.square(bond_price_err[(~self.large_err) & (np.array(self.bond_maturities) > 1.0)])))
        self.bond_prices_out = np.vstack((self.bond_prices, self.cleanPrices))
        self.nss_params_out = self._params()
        self.end_criterion = self.yieldcurve.fitResults().endCriterion()
        self.min_value = self.yieldcurve.fitResults().minimumCostValue()

    def _assign_params(self, nss_params):
        NSS = ql.SvenssonFitting(Quantlib_Helpers.ql_array(np.ones((len(self.bonds),)) / float(len(self.bonds))))
        self.yieldcurve = ql.FittedBondDiscountCurve(self.obs_date_ql, self.bond_helpers, self.opt.day_counter,
                                                     NSS, 1.0, 0, Quantlib_Helpers.ql_array(nss_params), 1.0)
        self.discountingTermStructure.linkTo(self.yieldcurve)
        self.yieldcurve.enableExtrapolation()

    def _fit_yields(self, previous_solution, bounds):
        # Yield curve fitting parameters
        tolerance = 1.0e-8
        maxiter = 10000
        # Option 1: Cubic spline fitting
        # knotVector = [-3.0, -2.0, -1.0, 0.0, .25, .5, 2.0, 4.0, 5.0, 7.0, 10.0, 12.0, 13.0, 14.0]
        # BS = ql.CubicBSplinesFitting(knotVector, True)
        # yieldcurve = ql.FittedBondDiscountCurve(self.obs_date_ql, self.bond_helpers,
        #                                         self.opt.day_counter, BS, tolerance, maxiter)
        # Option 2: Nelson-Svensson-Siegel
        if self.equal_weight:
            NSS = ql.SvenssonFitting(Quantlib_Helpers.ql_array(np.ones((len(self.bonds),)) / float(len(self.bonds))))
        else:
            NSS = ql.SvenssonFitting()


        if self.fit_method is FitMethod.NELDER_MEAD:
            NSS.setOptimizer(ql.FittingMethod.DiscountFitSimplex)
        elif self.fit_method is FitMethod.DE:
            NSS.setOptimizer(ql.FittingMethod.DiscountFitDiffEvo)
            NSS.setDifferentialEvolutionParams(60, 0.5, 0.9,
                                               ql.DifferentialEvolution.CurrentToBest2Diffs,
                                               # ql.DifferentialEvolution.BestMemberWithJitter,
                                               ql.DifferentialEvolution.Normal, 5, True, False)
        elif self.fit_method is FitMethod.ANNEALING:
            maxiter = 5000
            NSS.setOptimizer(ql.FittingMethod.DiscountFitAnnealing)
            NSS.setSimulatedAnnealingParams(initialTemp = 10, finalTemp_ = .00001,
                                            resetSteps_ = 150, reAnnealSteps_ = 50)
        else:
            raise Exception('No optimization method was given.')

        if self.regularize_prior is not None:
            NSS.setRegularization(0.0001, Quantlib_Helpers.ql_array(self.regularize_prior))

        if bounds is not None and hasattr(NSS, 'setBounds'):
            NSS.setBounds(*[Quantlib_Helpers.ql_array(list(b)) for b in bounds])

        # TODO: make this a toggle instead of specifying by bounds = None
        if bounds is None:
            NSS.useConstraints(False)
        else:
            NSS.useConstraints(True)

        if previous_solution is None:
            previous_solution = _nullSolution

        self.yieldcurve = ql.FittedBondDiscountCurve(self.obs_date_ql,
                                                     self.bond_helpers,
                                                     self.opt.day_counter,
                                                     NSS, tolerance, maxiter,
                                                     Quantlib_Helpers.ql_array(previous_solution), 1.0, 200)

        self.discountingTermStructure.linkTo(self.yieldcurve)
        self.yieldcurve.enableExtrapolation()

    def par_yields(self):
        # Move away from end-of-month shenanigans?
        # firstDate = ql.Date_endOfMonth(self.obs_date_ql) + ql.Period(5, ql.Days)
        firstDate = ql.Date_nextWeekday(self.obs_date_ql, ql.Wednesday)
        dates = [firstDate + ql.Period(m, ql.Months) for m in range(1, 121, 2)]
        dc = ql.SimpleDayCounter()
        zc_yield = [self.yieldcurve.zeroRate(d, self.yieldcurve.dayCounter(), ql.Continuous).rate() for d in dates]
        zc = ql.CubicZeroCurve([self.obs_date_ql] + dates, np.hstack((zc_yield[0], zc_yield)), dc, ql.NullCalendar())
        zc.enableExtrapolation()
        par_yield = np.array([Quantlib_Helpers.par_yield_semiannual(zc, d) for d in dates])
        # par_yield = np.array([Quantlib_Helpers.par_yield_semiannual(self.yieldcurve, d) for d in dates])

        return par_yield, dates

    def plot_par_curve_fit(self, split_bonds=False):
        # Make plots of yield to maturity.
        # Why is par_yields so damn finicky??
        par_yield, dates = self.par_yields()
        yield_plot_mats = np.array([self.opt.day_counter.yearFraction(self.obs_date_ql, d) for d in dates])
        fig, axes = plt.subplots()
        plt.plot(self.bond_plot_mats, self.bond_plot_yields * 100, 'b+')
        if split_bonds:
            plt.plot(yield_plot_mats, par_yield * 100, 'k')
            # plt.plot(yield_plot_mats[yield_plot_mats > 1.0], par_yield[yield_plot_mats > 1.0] * 100, 'r')
        else:
            plt.plot(yield_plot_mats, par_yield * 100, 'k')
        if self.bad_bond_info is not None:
            plt.plot(self.bad_bond_info['plot_mats'], self.bad_bond_info['bondYields'] * 100, 'r^')
        axes.set_xlim(0.0, 10.0)
        # axes.set_xlim(1.0, 10.0)
        return axes
        # If you want to plot date_mat on x axis:
        # axes.xaxis.set_major_locator(mdates.AutoDateLocator())
        # axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        # fig.autofmt_xdate()

    def plot_fit_prices(self):
        """ Test fitting."""
        fig, axes = plt.subplots()
        axes.plot(self.bond_plot_mats, self.bond_prices, 'k.')
        axes.plot(self.bond_plot_mats, self.cleanPrices, 'b.')
        plt.legend("Observed", "Fitted")

    def plot_fit_yields(self):
        """ Test fitting."""
        fig, axes = plt.subplots()
        axes.plot(self.bond_plot_mats, self.bond_yields_observed, 'k.')
        axes.plot(self.bond_plot_mats, self.bond_yields[:, 1], 'b.')
        plt.legend("Observed", "Fitted")








class FitZeroCurveQuantlib:
    import Quantlib_Helpers
    opt = Quantlib_Helpers.QuantlibFitOptions()
    yieldcurve = None
    discountingTermStructure = None
    bondPricingEngine = None
    obs_date_ql = None

    def __init__(self, zc_yields, dates, obs_date):
        """
        :param zc_yields:  Continuously compounded yields at specified dates in the future.
        :param obs_date:
        """
        self.obs_date_ql = Quantlib_Helpers.timestamp_to_qldate(obs_date)
        dates = Quantlib_Helpers.timestamp_to_qldate(dates)
        self.yieldcurve = ql.ZeroCurve([self.obs_date_ql] + dates, np.hstack((zc_yields[0], zc_yields)), self.opt.day_counter, self.opt.calendar)
        self.yieldcurve.enableExtrapolation()
        self.discountingTermStructure = ql.YieldTermStructureHandle(self.yieldcurve)
        self.bondPricingEngine = ql.DiscountingBondEngine(self.discountingTermStructure)

    # Note: maturity is time in days
    # tdatdt is CRSP format. We don't pass quantlib stuff through here!
    def fit_bond_to_curve(self, price, coupon, maturity, tdatdt):
        issue_date = Quantlib_Helpers.timestamp_to_qldate(tdatdt)
        maturity_date = self.obs_date_ql + maturity
        _, bond = _make_bond(self.opt, price, coupon, issue_date, maturity_date)

        # WTF, bond pricing engine not working??? We can calculate NPV by hand...
        # fixed_rate_bond.setPricingEngine(self.bondPricingEngine)
        # p_fitted = fixed_rate_bond.cleanPrice()
        p_dirty = ql.CashFlows_npv(bond.cashflows(), self.discountingTermStructure, False, self.obs_date_ql)
        p_fit = p_dirty - bond.accruedAmount()

        # Compute observed ytm and ytm for the fitted price
        y_obs = _yield_for_bond(bond, price, coupon, self.obs_date_ql, self.opt)
        y_fit = _yield_for_bond(bond, p_fit, coupon, self.obs_date_ql, self.opt)

        # del fixed_rate_bond
        return price, p_fit, y_obs, y_fit


