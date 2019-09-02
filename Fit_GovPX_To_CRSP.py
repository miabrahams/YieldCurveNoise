import pandas as pd
import numpy as np
from tqdm import tqdm
from mga_helpers import *
from CRSP_Helpers import *
import matplotlib.pyplot as plt
import QuantLib as ql
import Quantlib_Helpers
import CRSPQuantlib
import importlib



importlib.reload(CRSPQuantlib)


# Load data
CRSP_by_cusip = loadpklz('data/CRSP_by_cusip.pklz')
CRSP_by_cusip_useful = CRSP_by_cusip[['TMATDT', 'ITYPE', 'TCOUPRT', 'TDATDT']]


#On-the-run cusips from GovPX
OnTheRunCusips:pd.DataFrame = loadpklz('data/Output/on_the_run_cusips.pklz')
OnTheRunCusips = OnTheRunCusips.merge(CRSP_by_cusip_useful, left_on='cusip', right_index=True)
OnTheRunCusips['maturity'] = (OnTheRunCusips.TMATDT - OnTheRunCusips.date).dt.days.as_matrix()

#GovPX estimates
GovPXResults  = loadpklz('data/temp/GovPX_fit_results_refined.pklz')
GovPXDates  = sorted(list(GovPXResults.keys()))
GovPXResults_list = [GovPXResults[d] for d in GovPXDates]
GovPX_yieldcurve_output = [(r['yieldcurve']) for r in GovPXResults_list]
GovPX_fit_err_std_raw = np.array([r['fit_err'] for r in GovPXResults_list])
GovPX_fit_err_std = pd.Series(GovPX_fit_err_std_raw, index= GovPXDates)


f = GovPX_fit_err_std.plot() # Before
f.lines[0].set_linewidth(0.5)


rolling_mean = GovPX_fit_err_std.rolling(3, center=True, min_periods=1).mean()
# (GovPX_fit_err_std - rolling_mean).clip(lower=0).plot()
spike = (GovPX_fit_err_std - rolling_mean) > 0.00025
GovPX_fit_err_std.index[spike]
correction = (GovPX_fit_err_std.shift(-1) + GovPX_fit_err_std.shift(1)) / 2
# correction.plot()
GovPX_fit_err_std[spike] = correction[spike]

# f = GovPX_fit_err_std.plot() # After
f = GovPX_fit_err_std.resample('1W').mean().plot() # After
f.lines[0].set_linewidth(0.5)




# GSW baseline
gsw_params: pd.DataFrame = loadpklz("data/daily_gsw_params.pklz")
gsw_params['beta1'].plot()
plt.title("beta_0 in the GSW dataset")

# CRSP estimates
CRSPResults = loadpklz('data/temp/CRSP_fit_results_refined.pklz')
CRSPdates = sorted(list(CRSPResults.keys()))
CRSPResults_list = [CRSPResults[d] for d in CRSPdates]

CRSP_yieldcurve_output = [(r['yieldcurve']) for r in CRSPResults_list]
CRSP_fit_err_std_raw = np.array([r['fit_err'] for r in CRSPResults_list])
# cusips           = [r[1] for r in CRSPResults_OK]
# cleanPrices      = [r[2] for r in CRSPResults_OK]
# cleanPrices      = [r[2] for r in CRSPResults_OK]
# bond_prices_out  = [r[3] for r in CRSPResults_OK]

CRSP_fit_err_std = pd.Series(CRSP_fit_err_std_raw, index=CRSPdates)
rolling_mean = CRSP_fit_err_std.rolling(3, center=True, min_periods=1).mean()
(CRSP_fit_err_std - rolling_mean).clip(lower=0).plot()

CRSP_fit_err_std.plot()

f = CRSP_fit_err_std.plot() # Before
f.lines[0].set_linewidth(0.5)

spike = (CRSP_fit_err_std - rolling_mean) > 0.00025
CRSP_fit_err_std.index[spike]
# ['1996-04-05', '1998-01-23', '1998-07-10', '1998-07-21', '1998-10-07', '1999-01-28', '2006-06-30', '2011-08-01', '2011-08-03', '2011-08-05'],
correction = (CRSP_fit_err_std.shift(-1) + CRSP_fit_err_std.shift(1)) / 2
correction.plot()
CRSP_fit_err_std[spike] = correction[spike]
CRSP_fit_err_std.plot()

# CRSP_fit_err_std.rolling(3, center=True, min_periods=1, win_type='triang').sum().plot()





CRSP_fit_err_std.plot()
quickplot(CRSP_fit_err_std.resample('1W').mean())
quickplot(CRSP_fit_err_std.resample('1W').last())



importlib.reload(CRSPQuantlib)

GovPX_obs_dates = np.sort(OnTheRunCusips.date.unique())
OnTheRunResults = []
for d in tqdm(GovPX_obs_dates):
    CRSP_match = np.argwhere(CRSPdates == d)
    if len(CRSP_match) != 1:
        print("Couldn't match date " + str(d))
        continue
    CRSP_yields_matched = CRSP_yieldcurve_output[int(CRSP_match)]
    continuous_yields = np.log(1.0 + CRSP_yields_matched[:,2])
    zc_fitter = CRSPQuantlib.FitZeroCurveQuantlib(continuous_yields, d)
    OnTheRunToday = OnTheRunCusips[OnTheRunCusips.date == d]
    for i in OnTheRunToday.itertuples():
        fit_results = zc_fitter.fit_bond_to_curve(i.price, i.TCOUPRT, i.maturity, i.TDATDT)
        # TODO: be more careful dropping things
        if abs(fit_results[3] - fit_results[2]) < .01:
            OnTheRunResults.append([i.cusip, i.date, i.TCOUPRT] + list(fit_results))


OnTheRunResults = pd.DataFrame(OnTheRunResults, columns=['CUSIP', 'date', 'coupon', 'price', 'p_fitted', 'y_obs', 'y_fitted'])
OnTheRunResults['y_prem'] = OnTheRunResults.y_fitted - OnTheRunResults.y_obs
# OnTheRunResults.y_prem = np.fmax(OnTheRunResults.y_prem, 0)


#Number of observations
OnTheRunResults.groupby('date').CUSIP.count().plot()


# Plot y_prem
Premium = OnTheRunResults.groupby('date')['y_prem'].mean()
# Premium = np.fmax(Premium, 0)
Premium.plot()
Premium.resample('1M').mean().plot()
Premium.resample('1M').last().plot()

Premium = OnTheRunResults.groupby('date')['y_prem'].mean()


# Look at error for bills only
OTR_Bills = OnTheRunResults[OnTheRunResults.coupon == 0]

bill_premium = OTR_Bills.groupby('date').y_prem.mean()
bill_premium.plot()

bad_date = bill_premium.loc[bill_premium > .006].index[1].to_datetime64()

OnTheRunResults.loc[OnTheRunResults.date == bad_date]

i = OnTheRunCusips[OnTheRunCusips.date == bad_date].loc[1]
fit_results = zc_fitter.fit_bond_to_curve(i.price, i.TCOUPRT, i.maturity, i.TDATDT)


CRSP_match = np.argwhere(CRSPdates == bad_date)
len(CRSP_match)
CRSP_yields_matched = CRSP_yieldcurve_output[int(CRSP_match)]
quickplot(CRSP_yields_matched[:,2], CRSP_yields_matched[:,0])
zc_fitter = CRSPQuantlib.FitZeroCurveQuantlib(np.log(1.0 + CRSP_yields_matched[:,2]), bad_date)
OnTheRunResults_tmp = []
for i in OnTheRunToday.itertuples():
    fit_results = zc_fitter.fit_bond_to_curve(i.price, i.TCOUPRT, i.maturity, i.TDATDT)
    OnTheRunResults_tmp.append([i.cusip, i.date, i.TCOUPRT] + list(fit_results))




CRSPQuantlib.FitZeroCurveQuantlib()

price = i.price
coupon = i.TCOUPRT
maturity = i.maturity
tdatdt = i.TDATDT
issue_date = Quantlib_Helpers.timestamp_to_qldate(tdatdt)
maturity_date = zc_fitter.obs_date_ql + maturity
_, fixed_rate_bond = CRSPQuantlib._make_bond(zc_fitter.opt, price, coupon, issue_date, maturity_date)

#Remember to account for coupon freq here
# Delete this

zc_fitter.opt.day_counter.yearFraction(zc_fitter.obs_date_ql, maturity_date)
99.455 = 100*(1+y)^(-.24)

import math
math.exp(-math.log(.99455)*4)

zc_fitter.yieldcurve.referenceDate()
fixed_rate_bond.setPricingEngine(empty_engine)
zc_fitter.yieldcurve.dates()
p_fitted = fixed_rate_bond.cleanPrice()


p_dirty = ql.CashFlows_npv(fixed_rate_bond.cashflows(), zc_fitter.discountingTermStructure, False, zc_fitter.obs_date_ql)
p_fitted = p_dirty - fixed_rate_bond.accruedAmount()
fixed_rate_bond.dirtyPrice()



tdatdt[current_obs]


