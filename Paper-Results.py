from tqdm import tqdm
from mga_helpers import *
from CRSP_Helpers import *
from Quantlib_Helpers import *
import matplotlib.pyplot as plt
import CRSPQuantlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import matplotlib as mpl

import importlib
importlib.reload(CRSPQuantlib)

# Output options
import matplotlib.style as mplstyle
mplstyle.use('fast')
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

fig, ax = plt.subplots()
fig = plt.gcf()








# region Load data

# Throw away these dates. Careful!
spike_dates = pd.Series(pd.to_datetime(['1992-10-29', '1998-01-23', '1998-07-10', '1998-07-21', '1999-01-28', '2006-06-30', '2013-11-22']))
# NOTE: throwing away that day in 1992 is because of a single bad fit in the EW estimate, fix this later

# Static per-bond info
CRSP_by_cusip = loadpklz('data/CRSP_by_cusip.pklz')
CRSP_by_cusip_useful = CRSP_by_cusip[['TMATDT', 'ITYPE', 'TCOUPRT', 'TDATDT']]

# On-the-run cusips (from GovPX)
OnTheRunCusips:pd.DataFrame = loadpklz('data/Output/on_the_run_cusips.pklz')
OnTheRunCusips = OnTheRunCusips.merge(CRSP_by_cusip_useful, left_on='cusip', right_index=True)
OnTheRunCusips['maturity'] = (OnTheRunCusips.TMATDT - OnTheRunCusips.date).dt.days.as_matrix()

# GSW baseline data
gsw_params: pd.DataFrame = loadpklz("data/daily_gsw_params.pklz")

# HPW noise measure
HPW_Noise:pd.DataFrame = pd.read_pickle('data/HPW/HPW.pickle')

HPW_Noise['Noise_bp_rolling'] = HPW_Noise['Noise_bp'].rolling(3, min_periods=1).mean()
HPW_Noise['Noise_bp_rolling'].iloc[0] = HPW_Noise['Noise_bp_rolling'].iloc[1]

AvgBondInfo = pd.read_pickle("data/AvgBondInfo.pklz", compression="gzip")



# Spike ID - only needs to be done once
DO_SPIKE_ID = False
if DO_SPIKE_ID:
    (dates, CRSP_list) = loadpklz('data/temp/CRSP_fit_results_list_refined.pklz')
    CRSP_df = pd.DataFrame({"fit_err": [r['fit_err']*10000 for r in CRSP_list], "price_err": [r['price_err'] * 100 for r in CRSP_list]}, index=dates)
    plt.close()
    plt.plot(CRSP_df.index.to_series(), CRSP_df.fit_err.as_matrix())
    plt.scatter(spike_dates.as_matrix(), CRSP_df.loc[spike_dates].fit_err.as_matrix(), color="r")
    CRSP_df.loc[spike_dates].fit_err.as_matrix()
    spike_dates.as_matrix()
    high_err_dates = np.array(CRSP_df[CRSP_df.price_err > .4].index)
    savepklz(high_err_dates, 'data/high_err_index.pklz')







# Load converged CRSP estimates
def load_CRSP_results_list(filename):
    (dates, CRSP_list) = loadpklz(filename)
    dates = [d for d in dates if not np.isin(d, spike_dates)]
    CRSP_list = [r for r in CRSP_list if not np.isin(r['obs_date'], spike_dates)]
    yc = [(r['yieldcurve']) for r in CRSP_list]
    CRSP_df = pd.DataFrame({"fit_err": [r['fit_err']*10000 for r in CRSP_list],
                            "price_err": [r['price_err'] for r in CRSP_list]},
                           index=dates)
    CRSP_df['fit_err_rolling'] = CRSP_df.fit_err.rolling(3, center=False, min_periods=1).mean()
    CRSP_df['price_err_rolling'] = CRSP_df.price_err.rolling(3, center=False, min_periods=1).mean()
    CRSP_df['log_price_err'] = np.log(CRSP_df.price_err)
    CRSP_df['log_price_err_rolling'] = np.log(CRSP_df.price_err_rolling)
    return dates, yc, CRSP_df, CRSP_list


# Unconstrained estimates (new version)
(CRSP_dates, CRSP_Unconstrained2_yc, CRSP_Unconstrained2, CRSP_Unconstrained2_Results_list) = load_CRSP_results_list('data/temp/CRSP_fit_results_list_refined.pklz')

# Unconstrained estimates (originals, bounds can go all over the place)
(CRSP_dates, CRSP_Unconstrained_yc, CRSP_Unconstrained, CRSP_Unconstrained_Results_list) = load_CRSP_results_list('data/Output/CRSP_fit_results_list_refined.pklz')


# Constrained estimates
(_, CRSP_Base_yc, CRSP_Base, CRSP_Base_Results_list) = load_CRSP_results_list('data/Constrained/CRSP_fit_results_list_refined.pklz')
# Weighted by price
(_, CRSP_EW_yc, CRSP_EW, CRSP_EW_Results_list) = load_CRSP_results_list('data/EqualWeight/CRSP_fit_results_list_refined.pklz')
# Unconstrained estimates, ridge
(_, CRSP_ZeroRidge_yc, CRSP_ZeroRidge, CRSP_ZeroRidge_Results_list) = load_CRSP_results_list('data/ZeroRidge/CRSP_fit_results_list_refined.pklz')
# Unconstrained EW estimates, ridge
(_, CRSP_EWZeroRidge_yc, CRSP_EWZeroRidge, CRSP_EWZeroRidge_Results_list) = load_CRSP_results_list('data/EWZeroRidge/CRSP_fit_results_list_refined.pklz')


# CRSP_Unconstrained2.loc[pd.Timestamp('5/19/1995')]
# CRSP_Unconstrained2.loc[pd.Timestamp('6/20/2003')]
# CRSP_Unconstrained2.loc[pd.Timestamp('5/29/2015')]

# ax.cla()
# ax.plot(CRSP_Unconstrained2.fit_err)
# CRSP_Unconstrained2.fit_err.quantile(.30)

# Old ridge regressions
# # Regularized daily
# (_, CRSP_Ridge_yc, CRSP_Ridge, CRSP_Ridge_Results_list) = load_CRSP_results_list('data/Regularized/CRSP_fit_results_list_refined.pklz')
# # Ridge regression, weighted by price
# (_, CRSP_EWRidge_yc, CRSP_EWRidge, CRSP_EWRidge_Results_list) = load_CRSP_results_list('data/RegularizedPrice/CRSP_fit_results_list_refined.pklz')


## Create a "champion estimate" from the minimum of various other estimates
all_ests = [CRSP_Unconstrained2, CRSP_EW, CRSP_ZeroRidge, CRSP_EWZeroRidge]
best_fit_errs = np.stack([f.fit_err for f in all_ests])
best_fit_errs[np.isnan(best_fit_errs)] = 10e6
best_price_errs = np.stack([f.price_err for f in all_ests])
best_price_errs[np.isnan(best_price_errs)] = 10e6
CRSP_Champion = pd.DataFrame({"fit_err": np.min(best_fit_errs, 0), "price_err": np.min(best_price_errs, 0)}, index=CRSP_dates)
CRSP_Champion['fit_err_best_unconstrained'] = pd.DataFrame({"fit_err": np.min(np.stack([f.fit_err for f in [CRSP_Unconstrained2, CRSP_ZeroRidge]]), 0)}, index=CRSP_dates)
CRSP_Champion['fit_err_unconstrained'] = CRSP_Unconstrained2.fit_err
CRSP_Champion['fit_err_ridge'] = CRSP_ZeroRidge.fit_err
CRSP_Champion['price_err_ridge'] = CRSP_EWZeroRidge.price_err
CRSP_Champion['log_price_err'] = np.log(CRSP_Champion['price_err'])
CRSP_Champion['fit_err_rolling'] = CRSP_Champion.fit_err.rolling(3, center=True, min_periods=1).mean().shift(-1)
CRSP_Champion['price_err_rolling'] = CRSP_Champion.price_err.rolling(3, center=True, min_periods=1).mean().shift(-1)
CRSP_Champion['log_price_err_rolling'] = CRSP_Champion.log_price_err.rolling(3, center=True, min_periods=1).mean().shift(-1)



# General comparison dataframe
CRSP_Comparison = CRSP_Unconstrained2.fit_err.to_frame()
CRSP_Comparison['fit_err_regularized'] = CRSP_ZeroRidge.fit_err
CRSP_Comparison['HPW'] = HPW_Noise['Noise_bp']
CRSP_Comparison['price_err_EW_unconstrained'] = CRSP_EW.price_err
CRSP_Comparison['log_price_err'] = CRSP_EW.price_err.apply(np.log)
CRSP_Comparison['price_err_EW_regularized'] = CRSP_EWZeroRidge.price_err


# Unused
# CRSP_Comparison['fit_err_constrained'] = CRSP_Base.fit_err
# CRSP_Comparison['fit_err_EW_unconstrained'] = CRSP_EW.fit_err
# CRSP_Comparison['fit_err_EW_regularized'] = CRSP_EWZeroRidge.fit_err


# HPW vs me
HPW_Versus_me = HPW_Noise
HPW_Versus_me = HPW_Versus_me.join(CRSP_Unconstrained2.fit_err)
HPW_Versus_me = HPW_Versus_me.join(CRSP_ZeroRidge.fit_err.rename('regularized_fit_err'))
HPW_Versus_me = HPW_Versus_me.join(CRSP_EW.price_err)
HPW_Versus_me = HPW_Versus_me.join(CRSP_EWZeroRidge.price_err.rename('regularized_price_err'))

HPW_Versus_me

# endregion










# region yields (Move down later)
yieldstack = [(np.stack([nss_yield(y['nss_params']) for y in res_list])) for res_list in
              [CRSP_Unconstrained2_Results_list, CRSP_ZeroRidge_Results_list, CRSP_EW_Results_list, CRSP_EWZeroRidge_Results_list]]

yieldstack = np.stack(yieldstack, axis=2)

ax.cla()
ax.plot(yieldstack[:,120,0])
ax.plot(yieldstack[:,120,1])
ax.plot(yieldstack[:,120,2])
ax.plot(yieldstack[:,120,3])

tenyear = np.squeeze(yieldstack[:,120,:])
ax.plot(np.std(yieldstack[:,[12, 24, 60, 119],:],2))

yield_mean = np.mean(yieldstack, 2)
yield_diff = yieldstack - yield_mean[:,:,np.newaxis]

ax.cla()
ax.plot(np.abs(yield_diff[:,120,3]))
ax.plot(np.arange(12,120), np.sqrt(np.mean(np.power(yield_diff[:,12:120,:], 2),0)))

mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'

# Add in GSW for comparison
CRSP_date_idx = pd.Series(np.arange(len(CRSP_dates))).rename('CRSP_idx').to_frame()
CRSP_date_idx.index = pd.DatetimeIndex(CRSP_dates)
gsw_params_merged = gsw_params.merge(CRSP_date_idx, left_index=True, right_index=True)
gsw_params_merged = gsw_params_merged.reindex(index = gsw_params_merged.index[::-1])
gsw_crsp_index = gsw_params_merged.CRSP_idx.as_matrix()
gsw_params_merged_mat = gsw_params_merged.as_matrix()[:,0:6]
gsw_yields = np.stack([nss_yield(gsw_params_merged_mat[d,:]) for d in range(gsw_params_merged_mat.shape[0])])


ax.cla()
ax.plot(gsw_params_merged.index, gsw_yields[:,60], linewidth=1)
ax.plot(gsw_params_merged.index, np.squeeze(yieldstack[gsw_crsp_index,60,:]), linewidth=1)
ax.set_ylabel("Yield", size="x-large")
ax.legend([ "GSW", "Benchmark", "Ridge", "Equal Weighted", "Equal Weighted Ridge"], loc='upper right', fontsize="large")
ax.set_title("Five year zero coupon yield", size="16")
fig.set_tight_layout(True)
plt.savefig(f"tabsandfigs/timeseries_fiveyear.pdf", format='pdf')




ax.cla()
ax.plot(gsw_params_merged.index, gsw_yields[:,120], linewidth=1)
ax.plot(gsw_params_merged.index, np.squeeze(yieldstack[gsw_crsp_index,120,:]), linewidth=1)
ax.set_ylabel("Yield", size="x-large")
ax.legend([ "GSW", "Benchmark", "Ridge", "Equal Weighted", "Equal Weighted Ridge"], loc='upper right', fontsize="large")
ax.set_title("Ten year zero coupon yield", size="16")
fig.set_tight_layout(True)
plt.savefig(f"tabsandfigs/timeseries_tenyear.pdf", format='pdf')


ax.cla()
ax.plot(np.arange(12,120), np.sqrt(np.mean(np.power(yield_diff[:,12:120,:] * 100, 2),0)), linewidth=1)
ax.set_ylabel("Root mean square err. (bps)", size="x-large")
ax.legend([ "Benchmark", "Ridge", "Equal Weighted", "Equal Weighted Ridge"], loc='upper left', fontsize="large")
ax.set_title("Deviation between estimates", size="16")
ax.set_xlabel("Maturity (months)", size="x-large")
fig.set_tight_layout(True)
plt.savefig(f"tabsandfigs/deviation.pdf", format='pdf')


ax.cla()
# gsw_yielddiff_merged = np.sqrt(np.mean(np.power(gsw_yields[:,12:120] - np.squeeze(yield_mean[gsw_crsp_index,12:120]),2), 0)) * 100
gsw_yielddiff_merged2 = np.sqrt(np.mean(np.power(gsw_yields[:,12:120,np.newaxis] - yieldstack[gsw_crsp_index,12:120,:],2), 0)) * 100
ax.plot(np.arange(12,120), gsw_yielddiff_merged2)
ax.set_ylabel("Deviation (basis points)", size="x-large")
ax.set_xlabel("Maturity (months)", size="x-large")
ax.legend([ "Benchmark", "Ridge", "Equal Weighted", "Equal Weighted Ridge"], loc='upper left', fontsize="large")
# ax.legend(["GSW comparison"], loc='upper left', fontsize="large")
ax.set_title("Mean square deviation from GSW estimates", size="16")
# fig.set_tight_layout(True)
plt.savefig(f"tabsandfigs/gsw_deviation.pdf", format='pdf')


# endregion













# region Summary stats for paper
np.mean(CRSP_Unconstrained2.fit_err)
np.std(CRSP_Unconstrained2.fit_err)


np.mean(CRSP_Unconstrained2.fit_err['2011-01-01':])
np.std(CRSP_Unconstrained2.fit_err['2011-01-01':])


np.mean(CRSP_EW.price_err)
np.std(CRSP_EW.price_err)
np.mean(CRSP_EW.price_err['2011-01-01':])
np.std(CRSP_EW.price_err['2011-01-01':])


number_bonds = np.array([len(r['cusips']) for r in CRSP_Unconstrained_Results_list])
np.arrays(number_bonds).mean()
np.argmin(number_bonds)
CRSP_Unconstrained2_Results_list[0].keys()

len(CRSP_by_cusip[CRSP_by_cusip.ITYPE < 5])
len(CRSP_by_cusip[CRSP_by_cusip.ITYPE == 4])
len(CRSP_by_cusip[CRSP_by_cusip.ITYPE == 2])
len(CRSP_by_cusip[CRSP_by_cusip.ITYPE == 1])


unconstrained_beta = np.array([r['nss_params'] for r in CRSP_Unconstrained2_Results_list])
ridge_beta = np.array([r['nss_params'] for r in CRSP_ZeroRidge_Results_list])

unconstrained_beta = np.array([r['nss_params'] for r in CRSP_EW_Results_list])
ridge_beta = np.array([r['nss_params'] for r in CRSP_EWZeroRidge_Results_list])

np.std(1.0 / ridge_beta, 0) / np.std(1.0 / unconstrained_beta, 0)
1.0 / ridge_beta
ridge_beta

ax.plot(unconstrained_beta[:,0] + unconstrained_beta[:,1])
ax.cla()
ax.plot(ridge_beta[:,0] + ridge_beta[:,1])

ax.plot(ridge_beta[:,0])
ax.plot(ridge_beta[:,1])


np.mean((CRSP_Champion.fit_err_best_unconstrained - CRSP_Champion.fit_err))
fit_diff = (CRSP_Champion.fit_err_best_unconstrained - CRSP_Champion.fit_err)


np.mean(CRSP_Base.fit_err - CRSP_Unconstrained2.fit_err)
np.std(CRSP_Base.fit_err - CRSP_Unconstrained2.fit_err)
np.std(CRSP_Base.fit_err) / np.std(CRSP_Unconstrained2.fit_err)
np.mean(CRSP_Unconstrained)
np.mean(CRSP_ZeroRidge.fit_err - CRSP_Unconstrained2.fit_err) / np.mean(CRSP_Unconstrained2.fit_err)



CRSP_Comparison.corr().as_matrix()

np.corrcoef(CRSP_Comparison.fit_err['2010/01/01':'2016/12/31'].resample('1W').mean(), CRSP_Comparison.HPW['2010/01/01':'2016/12/31'].resample('1W').mean())

CRSP_Comparison.fit_err['2010/01/01':'2016/12/31'].resample('1W').autocorr()
CRSP_Comparison.HPW['2010/01/01':'2016/12/31'].resample('1W').autocorr()

from statsmodels.tsa.seasonal import seasonal_decompose
sd = seasonal_decompose(AvgBondInfo.AvgMaturity.resample("1W").last())
dir(sd)
np.var(sd.seasonal) / np.var(AvgBondInfo.AvgMaturity.resample("1W").last())
plt.close()


# endregion









# CRSP_Champion.log_price_err.plot(ax=ax)
# CRSP_Unconstrained2.fit_err.plot(ax=ax)
# CRSP_ZeroRidge.fit_err.plot(ax=ax)
# CRSP_Base[CRSP_Base.fit_err < 28].fit_err.plot(ax=ax)


# plt.clf()
# ax = plt.subplot(1,1,1)
ax.cla()
CRSP_Comparison[['fit_err', 'fit_err_unconstrained', 'fit_err_regularized', 'fit_err_EW_regularized', 'fit_err_unconstrained']].plot(ax=ax)
CRSP_Comparison[['price_err_unconstrained', 'price_err_EW_unconstrained', 'price_err_EW_regularized']].plot(ax=ax)




########    ACTUAL PLOTS FOR EXPORT START HERE #########

from matplotlib.transforms import Bbox
import matplotlib as mpl
mpl.rcParams['axes.titlepad'] = 14
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
plot_bbox = Bbox([[0.125, 0.20000000000000007], [0.9, 0.88]])


plt.clf()
ax = plt.subplot(1,1,1)

ax.cla()
CRSP_Comparison['fit_err'].plot(color='k', ax=ax, linewidth=1, linestyle='-')
CRSP_Comparison['fit_err_regularized'].plot(color=(0.3, 0.3, 1.0), ax=ax, linewidth=0.5, linestyle='--')
ax.set_ylabel("Noise (bps)", size="x-large")
# ax.set_xlabel("Date", size="large")
ax.legend(["Standard estimate", "Regularized estimate"], loc='upper left', fontsize="x-large")
ax.set_title(f"Yield curve noise", size="16")
# fig.set_tight_layout(True)
# ax.get_position()
ax.set_position(plot_bbox)
plt.savefig(f"tabsandfigs/yield_noise_comparison.pdf", format='pdf')


ax.cla()
CRSP_Comparison['price_err_EW_unconstrained'].plot(color=(1.0, 0.5, 0.5), ax=ax, linewidth=1, linestyle='-')
CRSP_Comparison['price_err_EW_regularized'].plot(color='k', ax=ax, linewidth=0.5, linestyle='--')
ax.set_ylabel("Pricing error (\$USD)", size="x-large")
ax.legend(["Standard estimate", "Regularized estimate"], loc='upper left', fontsize="large")
ax.set_title("Price noise", size="16")
ax.set_position(plot_bbox)
plt.savefig(f"tabsandfigs/price_noise_comparison.pdf", format='pdf')

# def absmax(array_like):
#     return array_like[np.argmax(np.abs(array_like))]


xlim_good = ax.get_xlim()

# Log price err vs yield err
fit_err_standardized = CRSP_Comparison['fit_err_regularized']
fit_err_standardized = (fit_err_standardized - fit_err_standardized.mean()) / fit_err_standardized.std()
log_price_err_standardized = CRSP_Comparison['price_err_EW_regularized'].apply(np.log)
log_price_err_standardized  = (log_price_err_standardized  - log_price_err_standardized.mean())  / log_price_err_standardized.std()

ax.cla()
fit_err_standardized.plot(color='k', ax=ax, linewidth=1, linestyle='-')
log_price_err_standardized.plot(color=(.3,.3,1.0), ax=ax, linewidth=1, linestyle='-')
ax.set_ylabel("Standard deviations", size="x-large")
ax.legend(["Yield noise", "$log($Price noise$)$"], loc='upper left', fontsize="large")
ax.set_title("Log Price Noise Comparison (standardized)", size="16")
ax.set_position(plot_bbox)
plt.savefig(f"tabsandfigs/yield_vs_log_price.pdf", format='pdf')


CRSP_Unconstrained2[pd.to_datetime("May 19, 1995")]
CRSP_Unconstrained2.iloc[1094]


ax.cla()
HPW_Versus_me['regularized_fit_err'].plot(ax=ax, color='k', linewidth=0.5, linestyle="-")
HPW_Versus_me['Noise_bp'].plot(ax=ax, c=(0.0, 0.5, 0.0), linewidth=1.0, linestyle=':')
ax.set_ylabel("Noise (bps)", size="x-large")
ax.legend([ "Benchmark estimate", "HPW (2013)"], loc='upper left', fontsize="large")
ax.set_title("Comparison with Hu, Pan and Wang (2013)", size="16")
ax.set_position(plot_bbox)
ax.set_xlim(xlim_good)
plt.savefig(f"tabsandfigs/hupanwang_comparison.pdf", format='pdf')
# ax.plot(HPW_Versus_me.index, HPW_Versus_me['fit_err'], 'k-', linewidth=0.5)




# Plot since 2011
import matplotlib.transforms as mtransforms

ax.cla()
CRSP_Comparison.fit_err['1/1/2011':].plot(color='k', ax=ax, linewidth=1, linestyle='-')
recent_dates = np.array(CRSP_Comparison.fit_err['1/1/2011':].index)
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
curr_ylim = ax.get_ylim()
curr_xlim = ax.get_xlim()
# 2011 debt crisis
shade_pd = np.logical_and(recent_dates > np.datetime64('2011-07-01'), recent_dates < np.datetime64('2011-08-01'))
ax.fill_between(recent_dates, 0, 1, where=shade_pd, facecolor='k', alpha=0.3, transform=trans)
# 2012 Eurozone Crisis
shade_pd = np.logical_and(recent_dates > np.datetime64('2012-05-12'), recent_dates < np.datetime64('2012-06-12'))
ax.fill_between(recent_dates, 0, 1, where=shade_pd, facecolor='k', alpha=0.3, transform=trans)
# 2013 debt crisis
shade_pd = np.logical_and(recent_dates > np.datetime64('2013-10-01'), recent_dates < np.datetime64('2013-11-01'))
ax.fill_between(recent_dates, 0, 1, where=shade_pd, facecolor='k', alpha=0.3, transform=trans)
# Shanghai market crash
shade_pd = np.logical_and(recent_dates > np.datetime64('2016-01-01'), recent_dates < np.datetime64('2016-02-01'))
ax.fill_between(recent_dates, 0, 1, where=shade_pd, facecolor='k', alpha=0.3, transform=trans)
# 2016 US Presidential Election
shade_pd = np.logical_and(recent_dates > np.datetime64('2016-11-08'), recent_dates < np.datetime64('2016-12-08'))
ax.fill_between(recent_dates, 0, 1, where=shade_pd, facecolor='k', alpha=0.3, transform=trans)
# 2017 Government Shutdown
shade_pd = np.logical_and(recent_dates > np.datetime64('2017-12-01'), recent_dates < np.datetime64('2018-01-01'))
ax.fill_between(recent_dates, 0, 1, where=shade_pd, facecolor='k', alpha=0.3, transform=trans)
ax.set_ylim(curr_ylim)
ax.set_ylabel("Noise (bps)", size="x-large")
ax.set_title(f"Recent yield curve noise developments", size="16")


ax.text(np.datetime64("2011-04-15"), 2.92, "2011\nDebt\nCrisis", horizontalalignment="center", fontsize=9)
ax.text(np.datetime64("2012-10-20"), 2.5, "Eurozone\nCrisis", horizontalalignment="center", fontsize=9)
ax.text(np.datetime64("2014-04-25"), 2.7, "2013\nDebt Crisis", horizontalalignment="center", fontsize=9)
ax.text(np.datetime64("2015-08-08"), 3.1, "Shanghai\nCrash", horizontalalignment="center", fontsize=9)
ax.text(np.datetime64("2017-03-15"), 3.0, "2016 \nElection", horizontalalignment="center", fontsize=9)
ax.text(np.datetime64("2017-08-08"), 2.6, "US Govt\nShutdown", horizontalalignment="center", fontsize=9)
# ax.set_position(plot_bbox)
fig.set_size_inches(10/1.2, 6/1.2)
fig.set_tight_layout(True)
ax.set_xlim([np.datetime64("2010-11-01"), np.datetime64("2018-02-01")])

plt.savefig(f"tabsandfigs/RecentDevelopments.pdf", format='pdf')



plt.close()
fig,ax = plt.subplots()


(np.log(CRSP_Champion.price_err.rolling(3).min())).plot(ax=ax)
rolling_log_clean = np.log(CRSP_Champion.price_err.rolling(3, min_periods=2).min())
rolling_log_clean.iloc[0] = rolling_log_clean[1]
CRSP_Champion['rolling_log_clean'] = (rolling_log_clean - rolling_log_clean.mean()) / rolling_log_clean.std()


# CRSP_Champion.rolling_log_clean.plot(ax)



# region GSW parameter estimates
gsw_params.beta3.iloc[2000:].plot(ax=ax)
gsw_params['lambda0'] =  1.0 / gsw_params['kappa0']
gsw_params['lambda1'] =  1.0 / gsw_params['kappa1']
gsw_params.kappa0.plot(ax=ax)
gsw_params.kappa1.plot(ax=ax)
gsw_params.lambda1.plot(ax=ax)
# plt.plot(gsw_params.lambda0)
# plt.show()
ax.cla()


# endregion





#region NSS parameter estimates

# # This is not "base" but rather constrained
# CRSP_params = pd.DataFrame([x['nss_params'] for x in CRSP_Base_Results_list], index=CRSP_dates,
#                            columns=['beta0', 'beta1', 'beta2', 'beta3', 'kappa0', 'kappa1'])
# CRSP_params.beta1.plot()
# # CRSP_params.kappa1.plot()


CRSP_params = pd.DataFrame([x['nss_params'] for x in CRSP_Unconstrained2_Results_list], index=CRSP_dates,
                           columns=['beta0', 'beta1', 'beta2', 'beta3', 'kappa0', 'kappa1'])
CRSP_params.beta1.plot()
# CRSP_params.kappa1.plot()



# We really want this to look good....
plt.clf()
CRSP_Ridge_params = pd.DataFrame([x['nss_params'] for x in CRSP_ZeroRidge_Results_list], index=CRSP_dates,
                           columns=['beta0', 'beta1', 'beta2', 'beta3', 'kappa0', 'kappa1'])
CRSP_Ridge_params.beta1.plot()

CRSP_Ridge_params.kappa1.plot()


# endregion



# region Construct rolling averages
# CRSP_fit_err_std = CRSP_Base.fit_err
# rolling_mean = CRSP_fit_err_std.rolling(3, center=True, min_periods=1).mean()
# (CRSP_fit_err_std - rolling_mean).clip(lower=0).plot()
# CRSP_fit_err_std.plot()
# f = CRSP_fit_err_std.plot() # Before
# f.lines[0].set_linewidth(0.5)
# quickplot(CRSP_fit_err_std.resample('1W').mean())
# quickplot(CRSP_fit_err_std.resample('1W').last())# spike = (CRSP_fit_err_std - rolling_mean) > 0.00025

# Locate spikes
# print(CRSP_fit_err_std.index[spike])
# ['1996-04-05', '1998-01-23', '1998-07-10', '1998-07-21', '1998-10-07', '1999-01-28', '2001-09-21', '2006-06-30', '2007-04-06', '2007-08-30', '2007-09-21', '2007-12-26', '2008-03-19', '2008-04-18', '2008-10-09', '2008-10-16', '2011-08-01', '2011-08-03', '2011-08-05'],
# correction = (CRSP_fit_err_std.shift(-1) + CRSP_fit_err_std.shift(1)) / 2
# correction.plot()
# CRSP_fit_err_std[spike] = correction[spike]
# CRSP_fit_err_std.plot()
# CRSP_fit_err_std.rolling(3, center=True, min_periods=1, win_type='triang').sum().plot()

# endregion



# Weekly stuff?
# CRSP_Ridge.fit_err.resample('1W').last().plot()
# CRSP_Base.fit_err.resample('1W').last().plot()
# CRSP_Champion.fit_err.resample('1W').last().plot()
# plt.clf()
# plt.close()









# region Currency portfolios

# Load data
FXReturns:pd.DataFrame = pd.read_pickle('data/FX/FXPortfolios.pklz', compression='gzip')
FF:pd.DataFrame = pd.read_pickle('data/FamaFrench/FF3Facs.pickle') # Fama-French factors
FXReturns = FXReturns.merge(FF, left_index=True, right_index=True)

# Currency portfolio excess returns
for i in range(1, 7): FXReturns['RxPortfolio' + str(i)] = FXReturns['Portfolio' + str(i)] - FXReturns['RF']


smf.ols('Portfolio1 ~ MktRx', data = FXReturns).fit().summary()
smf.ols('RxPortfolio1 ~ MktRx', data = FXReturns).fit().summary()
smf.ols('RxPortfolio2 ~ MktRx', data = FXReturns).fit().summary()
smf.ols('RxPortfolio6 ~ MktRx', data = FXReturns).fit().summary()


# Construct monthly series
from pandas.tseries.offsets import MonthEnd
FitErrMonthly:pd.DataFrame = HPW_Noise.merge(CRSP_Champion, left_index=True, right_index=True)
FitErrMonthly.loc[FitErrMonthly.index[0] - pd.Timedelta("2D")] = FitErrMonthly.iloc[0]
FitErrMonthly.sort_index(inplace=True)
FitErrMonthly = FitErrMonthly.resample("1M").last()
FitErrMonthly.index += MonthEnd(0)
FitErrMonthly['dNoise_HPW'] = FitErrMonthly['Noise_bp'].diff(1) / 100
# FitErrMonthly['dNoise_HPW_rolling'] = FitErrMonthly['Noise_bp_rolling'].diff(1)
FitErrMonthly['dNoise'] = FitErrMonthly['fit_err_unconstrained'].diff(1) / 100
FitErrMonthly['dNoise_ridge'] = FitErrMonthly['fit_err_ridge'].diff(1) / 100
FitErrMonthly['dPrice'] = FitErrMonthly['price_err'].diff(1) / 100
FitErrMonthly['dPrice_ridge'] = FitErrMonthly['price_err_ridge'].diff(1) / 100
FitErrMonthly['dLogPrice'] = FitErrMonthly['log_price_err'].diff(1) / 100
# FitErrMonthly['dNoise_rolling'] = FitErrMonthly['fit_err_rolling'].diff(1)
# FitErrMonthly['dPrice_rolling'] = FitErrMonthly['price_err_rolling'].diff(1)
# FitErrMonthly['dLogPrice_rolling'] = FitErrMonthly['rolling_log_clean'].diff(1)

# HPW_Monthly.dNoise.plot()
FXRegression:pd.DataFrame = FXReturns.merge(FitErrMonthly, left_index=True, right_index=True)
FXRegression.dropna(inplace = True)

# Don't drop stuff!
# FXRegression.drop(index= FXRegression.index[FXRegression.index > pd.Timestamp("01/01/2015")] )


print(smf.ols('RxPortfolio1 ~ dNoise + MktRx', data = FXRegression).fit().summary().as_latex())
print(smf.ols('RxPortfolio6 ~ dNoise + MktRx', data = FXRegression).fit().summary().as_latex())
print(smf.ols('RxPortfolio1 ~ dPrice + MktRx', data = FXRegression).fit().summary().as_latex())
print(smf.ols('RxPortfolio6 ~ dPrice + MktRx', data = FXRegression).fit().summary().as_latex())
print(smf.ols('RxPortfolio1 ~ dLogPrice + MktRx', data = FXRegression).fit().summary().as_latex())
print(smf.ols('RxPortfolio6 ~ dLogPrice + MktRx', data = FXRegression).fit().summary().as_latex())
print(smf.ols('RxPortfolio1 ~ dNoise_HPW + MktRx', data = FXRegression).fit().summary().as_latex())
print(smf.ols('RxPortfolio6 ~ dNoise_HPW + MktRx', data = FXRegression).fit().summary().as_latex())

print(smf.ols('RxPortfolio1 ~ dNoise + MktRx', data = FXRegression).fit().summary().tables[1])
print(smf.ols('RxPortfolio1 ~ dNoise + MktRx', data = FXRegression).fit().summary().tables[1].as_latex_tabular())



# Beta, beta_mkt, R2, alpha;
# t_beta t_b_mkt,   , t_alpha;

for est in ['dNoise', 'dNoise_ridge', 'dPrice', 'dPrice_ridge', 'dLogPrice', 'dNoise_HPW']:
    fit = smf.ols('RxPortfolio1 ~ ' + est + '.diff(1) + MktRx', data = FXRegression).fit()
    fit2 = smf.ols('RxPortfolio6 ~ ' + est + ' + MktRx', data = FXRegression).fit()
    print(f" & {fit.params[1]:.3f} & {fit.params[2]:.3f} & {fit.rsquared*100:.2f}\\% & {fit.params[0]:.3f}" + f" & {fit2.params[1]:.3f} & {fit2.params[2]:.3f} & {fit2.rsquared*100:.2f}\\% & {fit.params[0]:.3f} \\\\")
    print(f" & [{fit.tvalues[1]:.2f}] & [{fit.tvalues[2]:.2f}] & & [{fit.tvalues[0]:.2f}]" + f" & [{fit2.tvalues[1]:.2f}] & [{fit2.tvalues[2]:.2f}] & & [{fit2.tvalues[0]:.2f}] \\\\")


print(smf.ols('RxPortfolio1 ~ dNoise + MktRx', data = FXRegression).fit().summary().tables[1].as_latex_tabular())
print(smf.ols('RxPortfolio6 ~ dNoise + MktRx', data = FXRegression).fit().summary().tables[1].as_latex_tabular())
print(smf.ols('RxPortfolio1 ~ dPrice + MktRx', data = FXRegression).fit().summary().tables[1].as_latex_tabular())
print(smf.ols('RxPortfolio6 ~ dPrice + MktRx', data = FXRegression).fit().summary().tables[1].as_latex_tabular())
print(smf.ols('RxPortfolio1 ~ dLogPrice + MktRx', data = FXRegression).fit().summary().tables[1].as_latex_tabular())
print(smf.ols('RxPortfolio6 ~ dLogPrice + MktRx', data = FXRegression).fit().summary().tables[1].as_latex_tabular())
print(smf.ols('RxPortfolio1 ~ dNoise_HPW + MktRx', data = FXRegression).fit().summary().tables[1].as_latex_tabular())
print(smf.ols('RxPortfolio6 ~ dNoise_HPW + MktRx', data = FXRegression).fit().summary().tables[1].as_latex_tabular())



# smf.ols('RxPortfolio1 ~ dNoise_HPW_rolling', data = FXRegression).fit().summary().as_latex()
# smf.ols('RxPortfolio6 ~ dNoise_HPW_rolling', data = FXRegression).fit().summary().as_latex()
# smf.ols('RxPortfolio1 ~ dNoise_rolling', data = FXRegression).fit().summary()
# smf.ols('RxPortfolio6 ~ dNoise_rolling', data = FXRegression).fit().summary()

# smf.ols('RxPortfolio1 ~ dPrice_rolling', data = FXRegression).fit().summary()
# smf.ols('RxPortfolio6 ~ dPrice_rolling', data = FXRegression).fit().summary()

# smf.ols('RxPortfolio1 ~ dLogPrice_rolling', data = FXRegression).fit().summary()
# smf.ols('RxPortfolio6 ~ dLogPrice_rolling', data = FXRegression).fit().summary()



# A little plot here
FXRegression[['RxPortfolio1', 'RF']].plot()
FXRegression.mean() * 1200

# endregion






# region HPW versus my data



ax.cla()
ax.plot(HPW_Versus_me.index, HPW_Versus_me['Noise_bp'], c='0.7', linewidth=1.0)
ax.plot(HPW_Versus_me.index, HPW_Versus_me['fit_err'], 'k-', linewidth=0.5)
ax.plot(HPW_Versus_me.index, HPW_Versus_me['ridge_fit_err'], 'b-', linewidth=0.5)
ax.plot(HPW_Versus_me.index, HPW_Versus_me['unconstrained_fit_err'], 'r:', linewidth=0.5)


ax.cla()
ax.plot(HPW_Versus_me.index, HPW_Versus_me['fit_err'], c='0.7', linestyle='-', linewidth=0.5)
ax.plot(HPW_Versus_me.index, HPW_Versus_me['ridge_fit_err'], 'k-', linewidth=0.5)
ax.plot(HPW_Versus_me.index, HPW_Versus_me['unconstrained_fit_err'], 'b-', linewidth=0.5)



ax.cla()
ax.plot(HPW_Versus_me.index, HPW_Versus_me['fit_err'] - HPW_Versus_me['ridge_fit_err'], 'k-', linewidth=0.5)


ax.cla()
ax.plot(HPW_Versus_me.index, HPW_Versus_me['ridge_fit_err'] - HPW_Versus_me['unconstrained_fit_err'], 'k-', linewidth=0.5)
np.mean(HPW_Versus_me['ridge_fit_err'] - HPW_Versus_me['unconstrained_fit_err'])
np.std(HPW_Versus_me['ridge_fit_err'] - HPW_Versus_me['unconstrained_fit_err'])



# HPW_Versus_me[['Noise_bp', 'fit_err']].plot()
plt.close()


# endregion




# region NSS theory plots
N = np.arange(0.01, 10.0, .002)
def nss_hump(L):
    kt = N / L
    return (1.0 - np.exp(-kt)) / kt - np.exp(-kt)
l_range = np.arange(0.1, 5.5, 0.01, dtype='double')
l = 2
plt.plot(N, nss_hump(l))
plt.show()
# endregion







# region Compute On The Run premium

CRSP_long = pd.concat([pd.DataFrame({'date':  r['obs_date'], 'CUSIP': r['cusips'], 'y_prem': r['bond_yields'][:,0] - r['bond_yields'][:,1]}) for r in CRSP_Unconstrained2_Results_list])
CRSP_long = CRSP_long.merge(CRSP_by_cusip_useful, left_on='CUSIP', right_index=True)
CRSP_long.reset_index(inplace=True)
CRSP_long['maturity'] = (CRSP_long.TMATDT - CRSP_long.date) / pd.Timedelta(days=365)
CRSP_long['issue_maturity'] = (CRSP_long.TMATDT - CRSP_long.TDATDT) / pd.Timedelta(days=365)

def get_yprem_by_newest(i, title):
    ix = CRSP_long.loc[i].groupby('date').maturity.idxmax()
    s = pd.Series(list(CRSP_long.loc[ix].y_prem))
    s.index = pd.DatetimeIndex(CRSP_long.loc[ix].date)
    return s.sort_index().rename(title).to_frame()


is_tenyear = np.logical_and(CRSP_long['issue_maturity'] < 11, CRSP_long['issue_maturity'] > 9)
is_fiveyear = np.logical_and(CRSP_long['issue_maturity'] < 6, CRSP_long['issue_maturity'] > 4)
is_oneyear = np.logical_and(CRSP_long['issue_maturity'] < 1.5, CRSP_long['issue_maturity'] > .8)


# issue_mat = (CRSP_by_cusip_useful.TMATDT - CRSP_by_cusip_useful.TDATDT) / pd.Timedelta(days=365)
# ax.hist(list(CRSP_long['issue_maturity']))
# ax.cla()
# ax.hist(issue_mat[issue_mat < 2])


otr_prem = get_yprem_by_newest(is_tenyear, 'prem_10y')
otr_prem = otr_prem.join(get_yprem_by_newest(is_fiveyear, 'prem_5y'))
otr_prem = otr_prem * 100

CRSP_EW.price_err.to_frame().join(otr_prem['prem_10y']).corr()


ax.cla()
ax.plot(otr_prem)
# endregion




# region Old On The Run Premium: use daily GovPX identifiers
# GovPX_obs_dates = np.sort(OnTheRunCusips.date.unique())
#
# # Quick sanity check against the other section. Looks alright.
# print(CRSP_long.loc[idx_otr[115]])
# print(OnTheRunCusips[OnTheRunCusips.date == GovPX_obs_dates[1]])



# OnTheRunResults = []
# for d in tqdm(GovPX_obs_dates):
#     CRSP_match = np.argwhere(CRSP_dates == d)
#     if len(CRSP_match) != 1:
#         print("Couldn't match date " + str(d))
#         continue
#     CRSP_yields_matched = CRSP_Base_yc[int(CRSP_match)]
#     continuous_yields = np.log(1.0 + CRSP_yields_matched[:,2])
#     zc_fitter = CRSPQuantlib.FitZeroCurveQuantlib(continuous_yields, d)
#     OnTheRunToday = OnTheRunCusips[OnTheRunCusips.date == d]
#     for i in OnTheRunToday.itertuples():
#         fit_results = zc_fitter.fit_bond_to_curve(i.price, i.TCOUPRT, i.maturity, i.TDATDT)
#         # TODO: be more careful dropping things
#         if abs(fit_results[3] - fit_results[2]) < .01:
#             OnTheRunResults.append([i.cusip, i.date, i.TCOUPRT] + list(fit_results))
#
#
# OnTheRunResults = pd.DataFrame(OnTheRunResults, columns=['CUSIP', 'date', 'coupon', 'price', 'p_fitted', 'y_obs', 'y_fitted'])
# OnTheRunResults['y_prem'] = OnTheRunResults.y_fitted - OnTheRunResults.y_obs
# # OnTheRunResults.y_prem = np.fmax(OnTheRunResults.y_prem, 0)


# # Number of observations
# OnTheRunResults.groupby('date').CUSIP.count().plot()
# # Plot y_prem
# Premium = OnTheRunResults.groupby('date')['y_prem'].mean()
# # Premium = np.fmax(Premium, 0)
# Premium.plot()
# Premium.resample('1M').mean().plot()
# Premium.resample('1M').last().plot()
# Premium = OnTheRunResults.groupby('date')['y_prem'].mean()
# # Look at error for bills only
# OTR_Bills = OnTheRunResults[OnTheRunResults.coupon == 0]
# bill_premium = OTR_Bills.groupby('date').y_prem.mean()
# bill_premium.plot()

# Fit curves
# OnTheRunResults_tmp = []
# for i in OnTheRunToday.itertuples():
#     fit_results = zc_fitter.fit_bond_to_curve(i.price, i.TCOUPRT, i.maturity, i.TDATDT)
#     OnTheRunResults_tmp.append([i.cusip, i.date, i.TCOUPRT] + list(fit_results))


# endregion





# region The Dreaded Table 2

HPW_Versus_me.columns

# Merge CRSP and HPW with downloaded series
err_otr_merge = otr_prem.join(HPW_Versus_me[['Noise_bp', 'fit_err', 'regularized_fit_err', 'price_err', 'regularized_price_err']])
otr_prem_monthly = make_monthly(err_otr_merge.rolling(3, min_periods=1).min(), add_endpoints=[True, True])
# otr_prem_monthly = make_monthly(err_otr_merge, add_endpoints=[True, True])
BigTableDataframe=pd.read_pickle("data/BigCorrelationTable/BigTableDataframe.pklz", compression='gzip')
liqFactors = BigTableDataframe.join(otr_prem_monthly, how="outer")
differenced = liqFactors[['PS', 'ValueWeightedMKT']] # Some factors come differenced
liqFactors = liqFactors.diff(1)
liqFactors.update(differenced)
crisis_dummy = pd.Series(0, index=BigTableDataframe.index) # Add crisis dummies
crisis_dummy['2007-07-01':'2010-01-01'] = 1
liqFactors['crisis_dummy'] = crisis_dummy
wide_crisis_dummy = crisis_dummy
wide_crisis_dummy['2006-07-01':'2011-01-01'] = 1
liqFactors['wide_crisis_dummy'] = wide_crisis_dummy
del crisis_dummy, wide_crisis_dummy

liqFactors.iloc[70:,:]
len(otr_prem_monthly)

ax.cla()
# ax.plot(liqFactors.PS)
# ax.plot(liqFactors.fit_err)
# ax.plot(liqFactors.Noise_bp)
ax.plot(-otr_prem_monthly.prem_10y)

ax.plot(BigTableDataframe.GCRepo)
ax.plot(BigTableDataframe.BondVol)
ax.plot(BigTableDataframe.Baa_Aaa)

corrTableVars = ['fit_err', 'regularized_fit_err', 'price_err', 'regularized_price_err', 'treas3m', 'prem_5y', 'prem_10y', 'BondVol', 'GCRepo', 'Libor', 'Baa_Aaa', 'VIX', 'ValueWeightedMKT']
corrTableVars = ['fit_err', 'regularized_fit_err', 'price_err', 'regularized_price_err', 'PS']
c = liqFactors[corrTableVars].corr()
print(c)

BigTableDataframe.BondVol.mean()

dir(fits[0])

fits = [smf.ols(f"price_err ~ {c}", liqFactors).fit() for c in corrTableVars[4:]]
[str(f.summary().tables[1][2][4]) for f in fits]
print(fits[3].summary())

print(summaries[0].tables[0][0])

for c in corrTableVars:


print(c.to_latex())


# Available Factors:
# treas3m, Libor VIX PS ValueWeightedMKT Baa_Aaa, slope_10y, BondVol, GCRepo, prem_10y, prem_5y]
fac = 'GCRepo'
fac = 'treas3m + Libor + VIX + ValueWeightedMKT + Baa_Aaa + slope_10y + BondVol + PS'


corrTableVars = ['fit_err', 'regularized_fit_err', 'price_err', 'regularized_price_err', 'treas3m', 'prem_5y', 'prem_10y', 'BondVol', 'GCRepo', 'Libor', 'Baa_Aaa', 'VIX', 'ValueWeightedMKT']



s = smf.ols(f"fit_err ~ {fac}", liqFactors).fit(); print(s.summary())
s = smf.ols(f"fit_err ~ {fac}", liqFactors).fit(); print(s.summary())
smf.ols(f"fit_err_ridge ~ {fac}", liqFactors).fit().summary()
smf.ols(f"price_err ~ {fac}", liqFactors).fit().summary()
smf.ols(f"Noise_bp ~ {fac}", liqFactors).fit().summary()



smf.ols(f"fit_err ~ {fac} + I(crisis_dummy * {fac})", liqFactors).fit().summary()
smf.ols(f"Noise_bp ~ {fac} + I(crisis_dummy * {fac})", liqFactors).fit().summary()
smf.ols(f"price_err ~ {fac} + I(crisis_dummy * {fac})", liqFactors).fit().summary()

smf.ols(f"fit_err ~ {fac} + I(wide_crisis_dummy * {fac})", liqFactors).fit().summary()
smf.ols(f"Noise_bp ~ {fac} + I(wide_crisis_dummy * {fac})", liqFactors).fit().summary()
smf.ols(f"price_err ~ {fac} + I(wide_crisis_dummy * {fac})", liqFactors).fit().summary()


from sklearn.linear_model import ElasticNet
sk_mat = liqFactors[['fit_err', 'treas3m', 'prem_5y', 'prem_10y', 'BondVol', 'Libor', 'Baa_Aaa', 'VIX', 'ValueWeightedMKT']].dropna()
sk_mat = liqFactors[['fit_err', 'treas3m', 'prem_5y', 'BondVol', 'Libor', 'Baa_Aaa', 'VIX', 'ValueWeightedMKT']].dropna()
sk_mat = ((sk_mat - sk_mat.mean()) / sk_mat.std()).as_matrix()
en = ElasticNet(alpha=0.4, l1_ratio=0.5).fit(sk_mat[:,1:], sk_mat[:,0])
print(en.coef_)
print(en.get_params())



smf.ols(f"fit_err ~ prem_10y", liqFactors).fit().summary()
smf.ols(f"fit_err ~ prem_5y", liqFactors).fit().summary()
smf.ols(f"fit_err ~ prem_5y + crisis_dummy", liqFactors).fit().summary()


s = smf.ols(f"fit_err ~ prem_10y", otr_prem_monthly).fit(); print(s.summary())
s = smf.ols(f"fit_err ~ prem_5y", otr_prem_monthly).fit(); print(s.summary())
ax.cla()
ax.plot(otr_prem_monthly.index, s.fittedvalues)
ax.plot(otr_prem_monthly.fit_err)




print(smf.ols(f"fit_err ~ {fac}", liqFactors).fit().summary().tables[1].as_latex_tabular())

print(smf.ols('RxPortfolio1 ~ dNoise + MktRx', data = FXRegression).fit().summary().tables[1].as_latex_tabular())




ax.cla()
smf.ols(f"fit_err ~ {fac}", liqFactors).fit().summary()
idx = liqFactors[['fit_err', fac]].dropna()
ax.plot(idx.index, s.fittedvalues)
ax.plot(idx.index, idx.fit_err)



ax.cla()
ax.plot(liqFactors.fit_err)
ax.plot(liqFactors.fit_err['2007-06-01':'2009-08-01'])
ax.plot(CRSP_Unconstrained2.fit_err)
ax.plot(CRSP_Unconstrained2.fit_err['2007-07-01':'2010-01-01'])



for est in ['dNoise', 'dNoise_ridge', 'dPrice', 'dLogPrice', 'dNoise_HPW']:
    smf.ols(f"fit_err ~ {fac} + I(crisis_dummy * {fac})", liqFactors).fit().summary()
    smf.ols(f"Noise_bp ~ {fac} + I(crisis_dummy * {fac})", liqFactors).fit().summary()
    smf.ols(f"price_err ~ {fac} + I(crisis_dummy * {fac})", liqFactors).fit().summary()
    fit = smf.ols('RxPortfolio1 ~ ' + est + ' + MktRx', data = FXRegression).fit()
    fit2 = smf.ols('RxPortfolio6 ~ ' + est + ' + MktRx', data = FXRegression).fit()
    print(f" & {fit.params[1]:.3f} & {fit.params[2]:.3f} & {fit.rsquared*100:.2f}\\% & {fit.params[0]:.3f}" + f" & {fit2.params[1]:.3f} & {fit2.params[2]:.3f} & {fit2.rsquared*100:.2f}\\% & {fit.params[0]:.3f} \\\\")
    print(f" & [{fit.tvalues[1]:.2f}] & [{fit.tvalues[2]:.2f}] & & [{fit.tvalues[0]:.2f}]" + f" & [{fit2.tvalues[1]:.2f}] & [{fit2.tvalues[2]:.2f}] & & [{fit2.tvalues[0]:.2f}] \\\\")




# endregion

