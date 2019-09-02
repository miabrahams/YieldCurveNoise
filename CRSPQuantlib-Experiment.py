# The rest of the points are coupon bonds. We assume that the YTM given for the bonds are all par rates.
# So we have bonds with coupon rate same as the YTM.


from Quantlib_Helpers import *
import importlib
import QuantLib as ql
import pandas as pd
from mga_helpers import *
import CRSPQuantlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np
from CRSP_Helpers import *


CRSPData = pd.read_pickle("data/CRSP_Treasuries.pkl.compress", compression="gzip")
first_cusip_issue = pd.read_pickle("data/first_issue_date.pkl.compress", compression="gzip")
unique_dates = pd.read_pickle("data/CRSPUniqueDates.pkl.compress", compression="gzip")
CRSPData = clean_bad_bonds(CRSPData)
unique_dates = pd.Series([u for u in unique_dates if u not in bad_dates])
first_cusip_issue_dict = dict(zip(first_cusip_issue.index, first_cusip_issue.IssueDate))
# Gurkaynak, Sack and Wright starting values for parameters
gsw_params: pd.DataFrame = loadpklz("data/daily_gsw_params.pklz")
# Get everything ready for fast parameter passing
bond_maturities = CRSPData.ttm.as_matrix()
itype = CRSPData.ITYPE.as_matrix()
caldt = CRSPData.CALDT.as_matrix()
bond_rates = CRSPData.TCOUPRT.as_matrix()
bond_prices = CRSPData.TDNOMPRC.as_matrix()
cusips = CRSPData.TCUSIP.as_matrix()
unique_dates_np = pd.to_datetime(unique_dates).as_matrix()
cal_date = CRSPData.CALDT.astype('category')
bond_tdatdt = CRSPData.TDATDT.as_matrix()
# Stack everything into a single ndarray for speed. Stacking datetime64 together with other types doesn't work.
npdata = np.stack([bond_maturities, itype, bond_rates, bond_prices, cusips], axis=1)


##### TESTING

# Best so far
apr_03_best_params = ql.Array( [5.7502795124539426e-12, 4.922549407660499, 4.787747793861142, 19.42439599101258, 0.5003770453206803, 0.07538988176046008])
apr_05_best_params = ql.Array([-443.158, 448.276, 160.167, 1103.64, 0.197248, 0.0527979])
jul_10_best_params = ql.Array([0.324442, 4.21542, 6.00256, 14.3762, 1.44664, 0.186048])
jul_21_best_params = ql.Array([2.41441, 2.39558, 3.15241, 8.46661, 1.17268, 0.172212])


# Get data
bad_days = ['1996-04-05', # Meh, whatever
            '1998-07-10',
            '1998-07-21',
            '1998-10-07',
            '1999-01-28',
            '2001-09-21',
            '2006-06-30',
            '2007-04-06',
            '2007-08-30',
            '2007-09-21',
            '2007-12-26',
            '2008-03-19',
            '2008-04-18', '2008-10-09', '2008-10-16', '2011-08-01', '2011-08-03', '2011-08-05']

obs_date = pd.to_datetime('2005-07-27').to_datetime64()
npdata2 = npdata[cal_date == obs_date]
start_params = ql_array(gsw_params.loc[obs_date])

# Fit
importlib.reload(CRSPQuantlib)
rs_ql = CRSPQuantlib.FitSecuritiesQuantlib(npdata2[:, 0], npdata2[:, 1], npdata2[:, 2], npdata2[:, 3], npdata2[:,4], bond_tdatdt[cal_date == obs_date], obs_date)
# fit = rs_ql.fit(start_params)

# bestParams = rs_ql.yieldcurve.fitResults().solution()
bestParams = start_params

bounds = [ql.Array([ -500,   -30,  0,    0,  0,  0]),
          ql.Array([    5,  500, 300,  3000,  2, 2])]
rs_ql.check_problems = True
fit = rs_ql.fit(start_params, bounds)
print(fit['fit_err'])
print(fit['price_err'])
params = rs_ql.nss_params_out
hitBounds = any([abs(p - low) < 1e-6 or abs(p  - high) < 1e-6 for (p, low, high) in zip(params, bounds[0], bounds[1])])
if hitBounds:
    print("Hit parameter constraint. Check bounds.")


print(params)

rs_ql.nss_params_out


# Stuff for testing only
ax = rs_ql.plot_par_curve_fit()
ax.set_xlim(1.0, 10.0)
rs_ql.plot_fit_prices()
rs_ql.fit_err_std


rs_no_estimation = CRSPQuantlib.FitSecuritiesQuantlib(npdata2[:, 0], npdata2[:, 1], npdata2[:, 2], npdata2[:, 3], npdata2[:,4], bond_tdatdt[cal_date == obs_date], obs_date)
fit_no_estimation = rs_no_estimation.set_nss_params(start_params)
rs_no_estimation.plot_par_curve_fit()
print(fit_no_estimation['fit_err'])
print(fit_no_estimation['price_err'])


gsw_yield_today = nss_yield(start_params)
est_yield_today = nss_yield(rs_ql.nss_params_out)
pd.DataFrame({"GSW": gsw_yield_today, "Est": est_yield_today}).plot()



from random import shuffle

importlib.reload(CRSPQuantlib)
shuffle(unique_dates_np)

def restartable_date():
    for d in unique_dates_np:
        yield d

bounds_base = [np.array([  -3000,  -3000,  -3000,   -3000,   0,  0]),
               np.array([   3000,   2000,   3000,    3000,   20, 2])]

# Loop to check parameter
for obs_date in restartable_date():
    npdata2 = npdata[cal_date == obs_date]
    start_params = ql.Array(gsw_params.loc[obs_date].as_matrix().tolist())
    start_params = gsw_params.loc[obs_date].as_matrix()
    # Make sure start_params are inside the bounds at least
    bounds = [ql.Array(np.min(np.stack([bounds_base[0], start_params]), 0).tolist()),
              ql.Array(np.max(np.stack([bounds_base[1], start_params]), 0).tolist())]
    # Fit
    rs_ql = CRSPQuantlib.FitSecuritiesQuantlib(npdata2[:, 0], npdata2[:, 1], npdata2[:, 2], npdata2[:, 3], npdata2[:,4], bond_tdatdt[cal_date == obs_date], obs_date)
    # fit = rs_ql.fit(start_params)
    # bestParams = rs_ql.yieldcurve.fitResults().solution()
    bestParams = ql.Array(start_params.tolist())
    rs_ql.check_problems = True
    fit = rs_ql.fit(bestParams, bounds)
    print(fit['fit_err'])
    print(fit['price_err'])
    rs_ql.nss_params_out
    params = rs_ql.nss_params_out
    hitBounds = [abs(p - low) < 1e-6 or abs(p  - high) < 1e-6 for (p, low, high) in zip(params, bounds[0], bounds[1])]
    if any(hitBounds):
        print("Hit parameter constraint. Check bounds.")
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        print(np.stack([list(bounds[0]), list(params), list(bounds[1])]))
        break





# Don't do this
# USE_ORIGINAL_QL_BETA_SCALE = False
# if USE_ORIGINAL_QL_BETA_SCALE:
#     start_params = gsw_params.loc[obs_date].as_matrix()
#     start_params[0:4] = start_params[0:4] / 100.0
#     start_params = ql.Array(start_params.tolist())

































# Load main dataset
CRSPData = pd.read_pickle("data/CRSP_Treasuries.pkl.compress", compression="gzip")
ESTIMATE_LONG_END = False


# Note: the function prototypes cannot be given automatically.
# What you need to do is look up the class in the SWIG file and match it against the C++ QuantLib reference info.
# http://quantlib.org/reference/annotated.html
# http://quantlib.org/reference/class_quant_lib_1_1_zero_coupon_bond.html

opt = QuantlibFitOptions()

first_cusip_issue = pd.read_pickle("data/first_issue_date.pkl.compress", compression="gzip")
first_cusip_issue_dict = dict(zip(first_cusip_issue.index, first_cusip_issue.IssueDate))

# Load cross-section from LoadCRSPData.py
# df_export = pd.read_pickle("data/daily_cross_section.pkl.compress", compression="gzip")


# query_date = pd.to_datetime('2015-01-06')
# query_date = pd.to_datetime('2017-12-29')
query_date = pd.to_datetime('1994-02-01')
# query_date = pd.to_datetime('1994-06-02')
# query_date = pd.to_datetime('1999-03-04')
# query_date = CRSPData.CALDT[100000]

# Defined in LoadCRSPData.py
from CRSP_Helpers import *
df_export = get_df_inplace(CRSPData, query_date)
# df_export = get_df_inflation(CRSPData, query_date)

# Check issue dates
unique_CUSIPS = CRSPData.drop_duplicates('TCUSIP', 'last')
errors = 0
for r in unique_CUSIPS.itertuples():
    try:
        issue_date_treas_auction = timestamp_to_qldate(first_cusip_issue_dict[r.TCUSIP])
    except KeyError:
        errors += 1
        print("CUSIP not found: " + r.TCUSIP)
    issue_date_crsp = timestamp_to_qldate(r.TDATDT)
    if issue_date_treas_auction != issue_date_crsp:
        errors += 1
        print("Issue date disagreement: " + str(issue_date_treas_auction) + " vs " + str(issue_date_crsp))


if ESTIMATE_LONG_END:
    # Include >10Y
    drop = (df_export.ttm < 28) | ((df_export.ttm < 366) & (df_export.ITYPE != 4))
else:
    # Exclude bonds > 10y
    drop = (df_export.ttm < 28) | ((df_export.ttm < 366) & (df_export.ITYPE != 4)) | (df_export.ttm > 3700)
df_export = df_export[~drop]
df_export = df_export.reset_index()


# Bond info
bond_maturities = df_export.ttm
bond_maturity_date = df_export.TMATDT
caldt = df_export.CALDT
bond_maturity_date_ql = [timestamp_to_qldate(m) for m in bond_maturity_date]
ttm_ql = [ql.Period(i, ql.Days) for i in df_export.ttm]
cusips = df_export.TCUSIP # Cusips to match with Treasury auction data
bond_tdatdt = df_export.TDATDT
bond_rates = df_export.TCOUPRT.as_matrix()
bond_prices = df_export.TDNOMPRC.as_matrix()
accruedAmount_CRSP = df_export.TDACCINT.as_matrix()
# quickplot(bond_rates, bond_maturities)


# Constants describing the bond payment schedule
calc_date = timestamp_to_qldate(caldt[0])
ql.Settings.instance().evaluationDate = calc_date


# Need to declare these before we construct ql.fixedRateBonds
discountingTermStructure = ql.RelinkableYieldTermStructureHandle()
bondPricingEngine = ql.DiscountingBondEngine(discountingTermStructure)



# Accrued interest looks good!
# fig, axes = plt.subplots()
# axes.plot(accruedAmount_est, 'k', label="Estimated")
# axes.plot(accruedAmount_CRSP, 'b:', label='CRSP')





# Choose fitting method

# Fit the yield curve. (Be careful about extrapolation.)
tolerance = 1.0e-14
maxiter = 10000


# B-splines method.
# Params: knotVector, constrainAtZero,
# optional array of weights equal to number of bonds,
# optional array "l2" of parameter regularization penalties
# Need to add several knot points outside of target maturity range, two before, and two after.
# knotVector = [.25, .5, 1.0, 2.0, 3.0, 5.0, 7.0, 9.9]
# knotVector = [-20.0, -10.0, 0.0, .25, .5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0, 40.0]
if ESTIMATE_LONG_END:
    knotVector = [-10.0, -5.0, -1.0, 0.0, .5, 1.0, 2.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 31.0, 35.0, 40.0]
else:
    knotVector = [-3.0, -2.0, -1.0, 0.0, .25, .5, 2.0, 4.0, 5.0, 7.0, 10.0, 12.0, 13.0, 14.0]  # For 10Y
BS = ql.CubicBSplinesFitting(knotVector, False)
BS_yieldcurve = ql.FittedBondDiscountCurve(calc_date, bond_helpers, opt.day_counter, BS, tolerance, maxiter)
BS_yieldcurve.enableExtrapolation()
print(BS_yieldcurve.fitResults().numberOfIterations())


#Nelson-Svensson-Siegel method.
NSS = ql.SvenssonFitting()
NSS_yieldcurve = ql.FittedBondDiscountCurve(calc_date, bond_helpers, opt.day_counter, NSS, tolerance, maxiter)
NSS_yieldcurve.enableExtrapolation()
print(NSS_yieldcurve.fitResults().numberOfIterations())



yieldcurve = BS_yieldcurve
yieldcurve = NSS_yieldcurve

discountingTermStructure.linkTo(yieldcurve) # Pricing engine


# Fit again with mispriced bonds excluded. Good idea or no?


## Yield curve fit diagnostics
# day_counter.yearFraction(calc_date, bond_maturity_date_ql[10])
# yieldcurve.discount(calc_date + max(bond_maturities))
# yieldcurve.zeroRate(calc_date + 3 * ql.Years, opt.day_counter, ql.Compounded).rate()
# yieldcurve.zeroRate(calc_date + 9 * ql.Years, opt.day_counter, ql.Compounded).rate()
# yieldcurve.zeroRate(.3, ql.Compounded).rate()


## bond NPV calculations
print(str(bonds[100].cleanPrice()) + " vs " + str(bond_prices[100]))





# This is pretty jagged. Why?
if ESTIMATE_LONG_END:
    yield_ql_dates = [calc_date + ql.Period(i, ql.Months) for i in range(1, 360)]
else:
    yield_ql_dates = [calc_date + ql.Period(i, ql.Months) for i in range(1, 120)]
yield_plot_mats = np.array([opt.day_counter.yearFraction(calc_date, d) for d in yield_ql_dates])
zc_yield = np.array([yieldcurve.zeroRate(m, opt.compounding).rate() for m in yield_plot_mats])
quickplot(zc_yield, yield_plot_mats)
importlib.reload(Quantlib_Helpers)
par_yield = np.array(([Quantlib_Helpers.par_yield_semiannual(yieldcurve, m) for m in yield_ql_dates]))
quickplot(par_yield, yield_plot_mats)

# Some definitions for plots
crsp_yield_annualized = df_export.TDYLD.as_matrix()
bond_ql_dates = pd.DatetimeIndex(df_export.TMATDT).to_pydatetime() # pydatetime is best for matplotlib
bond_plot_mats = np.array([opt.day_counter.yearFraction(calc_date, Quantlib_Helpers.timestamp_to_qldate(m)) for m in bond_maturity_date])

## Make plots of yield to maturity
# Standard plot
fig, axes = plt.subplots()
# plt.plot(bond_plot_mats[bond_plot_mats > 1], ytm[bond_plot_mats > 1] * 100, 'b+')
plt.plot(bond_plot_mats, ytm * 100, 'b+')
plt.plot(yield_plot_mats, par_yield * 100, color='red')
# plt.plot(bond_plot_mats[bond_plot_mats < 1], 100 * ytm[bond_plot_mats < 1], 'k+')
# axes.set_ylim(3.5, 8.0)
if ESTIMATE_LONG_END:
    axes.set_xlim(0, 30.0)
else:
    axes.set_xlim(0, 10.0)
# For date_mat on x axis:
# axes.xaxis.set_major_locator(mdates.AutoDateLocator())
# axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# fig.autofmt_xdate()

# plt.axvline(pd.DatetimeIndex(df.CALDT).to_pydatetime()[0] + datetime.timedelta(20*365), color='r')
# plt.scatter(df['TDDURATN'] / 365.0, df['TDDURATN'] / 365.0)


# Anything nasty here?
df_export[(ytm * 100 < 6) & (bond_plot_mats > 4)]



# Problem: can't match the CRSP yield to maturity with my own calculations
# ytm_diff = crsp_yield_annualized*(365.25/365) - (ytm * 100)
# ytm_diff_relative = ytm_diff / crsp_yield_annualized
# fig, axes = plt.subplots()
# plt.plot(bond_plot_mats, ytm * 100, '.')
# plt.plot(bond_plot_mats, crsp_yield_annualized*(365.25/365), '.')


# Compare methods
plotpoints = np.arange(1, 9.5, 0.1)
parYieldCurve_NSS = np.array([Quantlib_Helpers.par_yield_semiannual(NSS_yieldcurve, m) for m in bond_maturity_date_ql])
parYieldCurve_BS = np.array([Quantlib_Helpers.par_yield_semiannual(BS_yieldcurve, m) for m in bond_maturity_date_ql])
# yieldEst_NSS = [NSS_yieldcurve.zeroRate(m, ql.Compounded).rate() for m in plotpoints]
# yieldEst_BS = [BS_yieldcurve.zeroRate(m, ql.Compounded).rate() for m in plotpoints]

fig, axes = plt.subplots()
plt.plot(bond_plot_mats, ytm, 'k.')
axes.plot(bond_plot_mats, parYieldCurve_NSS, 'r', label='Svensson')
axes.plot(bond_plot_mats, parYieldCurve_BS, 'b', label='Cubic')
axes.legend()
# axes.xaxis.set_major_locator(mdates.AutoDateLocator())
# axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# fig.autofmt_xdate()






# Test the output from our CRSPQuantlib function
# maturity_dates = [b.maturityDate() for b in bonds]
maturity_range = np.arange(0, yieldcurve.maxTime(), 1.0/12.0)
yieldcurve_discount = np.array([yieldcurve.discount(m) for m in maturity_range], dtype=float )
yieldcurve_zeros = np.array([yieldcurve.zeroRate(m, ql.Compounded).rate() for m in maturity_range], dtype=float)
yieldcurve_output = np.stack([maturity_range, yieldcurve_discount, yieldcurve_zeros], axis=1)


# Test fitting. (Implemented in CRSPQuantlib)
cleanPrice_est = np.array([b.cleanPrice() for b in bonds], dtype=float)
fig, axes = plt.subplots()
axes.plot(bond_plot_mats, cleanPrice_est, 'b.')
axes.plot(bond_plot_mats, bond_prices, 'k.')



