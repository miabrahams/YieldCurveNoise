
import numpy as np
import os
from Quantlib_Helpers import *
from tqdm import tqdm
from mga_helpers import *
from CRSP_Helpers import *
import matplotlib.pyplot as plt
import QuantLib as ql
import CRSPQuantlib
import importlib


importlib.reload(CRSPQuantlib)
# importlib.reload(Quantlib_Helpers)

# Load data.
CRSP_by_cusip = loadpklz('data/CRSP_by_cusip.pklz')
CRSP_by_cusip_useful = CRSP_by_cusip[['TMATDT', 'ITYPE', 'TCOUPRT', 'TDATDT']]

# You need to have run LoadGovPX already to get pklz.
GovPXFiles = [f for f in os.listdir('data\\GovPX') if f.endswith(".pklz")]
GovPXFiles = GovPXFiles[9:] + GovPXFiles[0:9] #Rearrange since 2000 is "GovPX00.zip"

GovPX = loadpklz("data/GovPX/" + GovPXFiles[-6])
obs_dates = GovPX.date.unique()


# Gurkaynak, Sack and Wright starting values for parameters
gsw_params: pd.DataFrame = loadpklz("data/daily_gsw_params.pklz")



# TODO: Let's figure out what's going wrong with these hmm?
# Also remember to exclude them later...
# bad_dates = [np.datetime64('1999-06-30T00:00:00.000000000')]

# More bad CUSIPs go here
# CRSP_by_cusip.loc['912795PE']
# GovPX.ix[GovPX.cusip == '912795PE'].askprc



# Collect bond prices at the end of each trading day
def get_eod(govpx_data, obs_date):
    GovPX_Today = govpx_data[govpx_data.date == obs_date]
    if GovPX_Today.shape[0] < 100:
        print("Very few observations on " + str(pd.to_datetime(obs_date).date()))
        return None
    # Anything with a quote greater than 15   but less than $50 is likely an error
    # (Quotes under 15 are for bills, reported as %yield. we would have to tweak this for a different time period.)
    # Should probably figure out some sort of warning system.
    EODHasBidprc = GovPX_Today[~pd.isna(GovPX_Today.bidprc) & ~((GovPX_Today.bidprc > 15) & (GovPX_Today.bidprc < 50))].groupby('cusip')
    EODBidPrice = EODHasBidprc['bidprc'].last()
    EODAskPrice = GovPX_Today[~pd.isna(GovPX_Today.askprc) & ~((GovPX_Today.askprc > 15) & (GovPX_Today.askprc < 50))].groupby('cusip')['askprc'].last()
    EODMid = (EODBidPrice + EODAskPrice) / 2
    EODMid = EODMid.loc[EODHasBidprc['bidprc'].count().reindex_like(EODMid) > 3]  # Don't use anything with very few obs
    EODSpread = GovPX_Today[~pd.isna(GovPX_Today.askprc) & ~(pd.isna(GovPX_Today.bidprc))].groupby('cusip').last()
    EODSpread = EODSpread['askprc'] - EODSpread['bidprc']
    # AllCusips = GovPX_Today.groupby('cusip')['bidprc'].last().to_frame()
    # AllCusips = AllCusips.merge(CRSP_by_cusip_useful, left_index=True, right_index=True) # Bond info from CRSP data
    # Checking out odd quotes...
    # OddCusip = '912810BU'
    # EODInfo.loc[OddCusip]
    # EODBidPrice.loc[OddCusip]
    # EODAskPrice.loc[OddCusip]
    # GovPX_Today[~pd.isna(GovPX_Today.bidprc) & (GovPX_Today.cusip == OddCusip)]
    # CRSP_by_cusip.loc[OddCusip]
    # Following two lines are a bad idea
    # EODInfo[pd.isna(EODInfo)] = EODBidPrice
    # EODInfo[pd.isna(EODInfo)] = EODAskPrice
    EODMid.dropna(inplace=True)
    #Construct dataframe
    EODInfo = EODMid.to_frame('bond_prices')
    EODInfo['active'] = EODHasBidprc['active'].last()  # On the run
    EODInfo['spread'] = EODSpread
    EODInfo['time'] = EODHasBidprc['time'].last()  # Quote time
    EODInfo = EODInfo.merge(CRSP_by_cusip_useful, left_index=True, right_index=True) # Bond info from CRSP data
    if EODInfo['ITYPE'].hasnans:
        print("Failed to match all cusipS.")
        EODInfo.dropna(inplace=True)
    # Optional
    EODBidYld = GovPX_Today[~pd.isna(GovPX_Today.bidprc) & ~(GovPX_Today.bidprc < 50)].groupby('cusip')['bidyld'].last()
    EODInfo = EODInfo.join(EODBidYld, how='left')
    # Drop bad bonds
    EODInfo['bond_maturities'] = (EODInfo['TMATDT'] - obs_date).dt.days.as_matrix()
    EODInfo['itype'] = EODInfo['ITYPE'].as_matrix()
    EODInfo['bond_rates'] = EODInfo['TCOUPRT'].as_matrix()
    EODInfo['bond_tdatdt'] = EODInfo['TDATDT']
    drop = ((EODInfo.bond_maturities < 28) | (EODInfo.bond_maturities > 3700) |
            ((EODInfo.bond_maturities < 366) & (EODInfo.itype != 4)) | (EODInfo.itype > 4))
    EODInfo = EODInfo[~drop]
    # Adjust bill prices on an actual/360 day basis
    is_bill = EODInfo.itype == 4
    EODInfo.loc[is_bill, 'bond_prices'] = 100 * (1 - (EODInfo.loc[is_bill, 'bond_prices'] / 100.0) * EODInfo.loc[is_bill, 'bond_maturities'] / 360.0)
    EODInfo.loc[is_bill, 'spread'] = np.maximum(- 100 * ((EODInfo.loc[is_bill, 'spread'] / 100.0) * EODInfo.loc[is_bill, 'bond_maturities'] / 360.0), 0)
    EODInfo = EODInfo[EODInfo.bond_prices > 50.0]
    return EODInfo


# Do computation on all files
final_result = []
on_the_run_all = []
n_days_dropped = []
daily_by_CUSIP = []
detailed_bond_info = []
nss_params_out = []

# PARTIAL = False
# FIRST = False
# if FIRST:
#     GovPXFilesIter = GovPXFiles[0:6]
# else:
#     GovPXFilesIter = GovPXFiles[7:18]

GovPXFilesIter = GovPXFiles

# GovPXFilesIter = [GovPXFiles[10]]

gsw_est_params = list(gsw_params.iloc[-1]) # Match GSW yields
last_params = gsw_est_params # Track previous results
bounds_base = [np.array([  -3000,  -3000,  -3000,   -3000,   0,  0]),
               np.array([   3000,   3000,   3000,    3000,   20, 2])]



for filename in GovPXFilesIter:
    print("____________ Loading ____________")
    print(filename)
    GovPX = loadpklz("data/GovPX/" + filename)
    obs_dates = GovPX.date.unique()
    zc_yields = []
    par_yields = []
    fit_errs = []
    price_errs = []
    spreads = []
    on_the_run = pd.DataFrame()
    obs_dates_success = []
    bad_cusips_daily = []
    n_bonds = []
    obs_times = []
    daily_bond_info = [] # We will pack this with stuff
    daily_nss_params = []

    ### Only estimate 30 days for debugging
    # obs_dates = obs_dates[1:30]
    # obs_date = obs_dates[0]
    ###
    for i, obs_date in tqdm(zip(range(len(obs_dates)), obs_dates)):
        EODInfo = get_eod(GovPX, obs_date)
        if EODInfo is None:
            continue
        try:
            gsw_est_params = list(gsw_params.loc[obs_date])
        except KeyError:
            print("GSW did not provide estimates on " + str(obs_date))
            continue
        fit_err_outcome = [100.0, 100.0]
        result_options = [None, None]
        nss_param_options = [gsw_est_params, last_params]
        bounds = [ql_array(np.min(np.stack([bounds_base[0], gsw_est_params, last_params]), 0).tolist()),
                  ql_array(np.max(np.stack([bounds_base[1], gsw_est_params, last_params]), 0).tolist())]
        for j in range(0, 2):
            try:
                result_options[j] = CRSPQuantlib.FitSecuritiesQuantlib(EODInfo.bond_maturities, EODInfo.itype, EODInfo.bond_rates, EODInfo.bond_prices, EODInfo.index, EODInfo.bond_tdatdt, obs_date)
                result_options[j].check_problems = True
                result_options[j].fit(nss_param_options[j], bounds)
                params = result_options[j].nss_params_out
                hitBounds = [abs(p - low) < 1e-6 or abs(p  - high) < 1e-6 for (p, low, high) in zip(params, bounds[0], bounds[1])]
                if any(hitBounds):
                    print("Hit parameter bounds on " + str(obs_date))
                    result_options[j].fit(nss_param_options[j])  # Unbounded
                fit_err_outcome[j] = result_options[j].fit_err_std
            except RuntimeError:
                print("Failed to fit with method " + str(j) + " on " + str(obs_date))
        if np.min(fit_err_outcome) > 1:
            print("Bad date on " + str(obs_date))
            continue
        res = result_options[np.argmin(fit_err_outcome)]
        last_params = res.nss_params_out
        daily_nss_params.append(last_params)
        # gsw_yield_today = Quantlib_Helpers.nss_yield(gsw_est_params)
        # est_yield_today = Quantlib_Helpers.nss_yield(last_params)
        # pd.DataFrame({"GSW": gsw_yield_today, "Est": est_yield_today}).plot()
        # print(last_params)
        # print(gsw_est_params)
        # print(null_params)
        # fit_err_outcome
        # res_final.plot_fit_yields()  # For diagnostics
        # Output variables
        zc_yields.append(res.yieldcurve_output[:, 2])
        par_yields.append(res.par_yields()[0])
        fit_errs.append(res.fit_err_std)
        price_errs.append(res.price_err_rmse)
        obs_dates_success.append(obs_date)
        n_bonds.append(res.bond_prices_out.shape[1])
        # Begin output of detailed info by cusip
        EODInfo_by_cusip = EODInfo.loc[res.cusips]
        EOD_Time = EODInfo.loc[res.cusips].time.astype('int64')
        mean_time = pd.to_datetime(EOD_Time.mean().astype('datetime64[ns]'))
        std_time = EOD_Time.std().astype('timedelta64[ns]').astype('timedelta64[m]')
        obs_times.append([mean_time, std_time])
        on_the_run_cusips = EODInfo.active.loc[EODInfo.active == 'A'].index.tolist()
        def _cusipStatus(c):
            if c in on_the_run_cusips:
                return 1
            elif c in res.bad_cusips:
                return -1
            return 0
        cusipStatus = np.array([_cusipStatus(c) for c in res.cusips])
        daily_bond_info.append({"cusip": res.cusips, "date": np.repeat(res.obs_date, res.cusips.shape),
                                "price": res.bond_prices,  "fitted_price": res.cleanPrices,
                                "yields": res.bond_yields, "status": cusipStatus,
                                "spread": EODInfo_by_cusip.spread, "time": EODInfo_by_cusip.time})
        # TODO: correlate time with stuff
        on_the_run = on_the_run.append(pd.DataFrame({"date": np.repeat(obs_date, len(on_the_run_cusips)),
                                                     "price": EODInfo.bond_prices.loc[EODInfo.active == 'A'].tolist(),
                                                     "cusip": on_the_run_cusips}))
        spreads.append([EODInfo.spread.mean(), EODInfo.spread.std()])
        if len(res.bad_cusips) > 0:
            bad_cusips_daily.append([obs_date, res.bad_cusips])
    # quickplot([x for x in fit_errs if x < .01])
    # quickplot([x for x in fit_errs])
    obs_dates_success = np.stack(obs_dates_success)
    zc_yield = pd.DataFrame(np.stack(zc_yields), columns=["zc_" + str(x) + "m" for x in range(1,121)], index=obs_dates_success)
    par_yield = pd.DataFrame(np.stack(par_yields), columns=["par_" + str(x) + "m" for x in range(1,121,2)], index=obs_dates_success)
    spread = pd.DataFrame(spreads, columns=["mean_spread", "std_spread"], index=obs_dates_success)
    N = pd.DataFrame(n_bonds, columns=["n_bonds"], index=obs_dates_success)
    obs_time = pd.DataFrame(obs_times, columns=["mean_time", "std_time"], index=obs_dates_success)
    price_err = pd.DataFrame({"price_err": np.array(price_errs)}, index=obs_dates_success)
    fit_err = pd.DataFrame({"fit_err": np.array(fit_errs)}, index=obs_dates_success)
    fit_output = fit_err.join(price_err).join(zc_yield).join(par_yield).join(spread).join(N).join(obs_time)
    # Handle daily list of bad cusips
    if len(bad_cusips_daily) > 0:
        bcd_dates = pd.Series([a[0] for a in bad_cusips_daily])
        bcd_list = [a[1].tolist() for a in bad_cusips_daily]
        bad_cusips_daily_out = pd.DataFrame({"bad_cusips": bcd_list}, index=bcd_dates)
        fit_output = fit_output.merge(bad_cusips_daily_out, 'left', left_index=True, right_index=True)
    else:
        fit_output['bad_cusips'] = None
    detailed_bond_info += daily_bond_info
    final_result.append(fit_output)
    on_the_run_all.append(on_the_run)
    nss_params_out.append(np.array(daily_nss_params))
    n_days_dropped.append(len(obs_dates) - len(obs_dates_success))


# if PARTIAL and not FIRST:
#     savepklz(pd.concat(final_result), "data/temp/GovPX_Daily_Result.pklz")
#     savepklz(pd.concat(on_the_run_all), "data/temp/on_the_run_cusips.pklz")
#     savepklz(pd.DataFrame({'GovPXFile': GovPXFiles, 'n_dropped': n_days_dropped}), "data/temp/n_days_dropped.pklz")
#     savepklz(detailed_bond_info, "data/temp/GovPX_daily_bond_info.pklz")
#
# if PARTIAL and not FIRST:
#     pd.concat(final_result).append(loadpklz("data/temp/GovPX_Daily_Result.pklz"))
#     pd.concat(on_the_run_all).append(loadpklz("data/temp/on_the_run_cusips.pklz"))
#     pd.concat(pd.DataFrame({'GovPXFile': GovPXFiles, 'n_dropped': n_days_dropped})).append(loadpklz("data/temp/n_days_dropped.pklz"))

savepklz(pd.concat(final_result), "data/output/GovPX_Daily_Result.pklz")
savepklz(pd.concat(on_the_run_all), "data/output/on_the_run_cusips.pklz")
savepklz(pd.DataFrame({'GovPXFile': GovPXFiles, 'n_dropped': n_days_dropped}), "data/output/n_days_dropped.pklz")
savepklz(detailed_bond_info, "data/output/GovPX_daily_bond_info.pklz")




# Rename columns
# extra_final_result = pd.concat(final_result)
# extra_final_result = extra_final_result.rename(index=str, columns={"mean": "mean_spread", "std": "std_spread", "avg_time": "mean_time"})
# extra_final_result.columns
# savepklz(extra_final_result, "data/output/GovPX_Daily_Result.pklz")
# extra_final_result.n_bonds.iloc[-300:].plot()




##### Other stuff here #####
final_result:pd.DataFrame = loadpklz('data/Output/GovPX_Daily_Result.pklz')

final_result.fit_err.plot()
final_result.fit_err.loc[final_result.fit_err < 0.01].plot()




# if len(bad_cusips_daily) > 0:
#     bad_cusips_daily = np.stack(bad_cusips_daily, 0).squeeze()
#     bad_cusips_daily = pd.DataFrame({"date": bad_cusips_daily[:,0], "ind": bad_cusips_daily[:,1], "cusip": pd.Series(bad_cusips_daily[:,2])})
#     bad_cusips_daily.cusip.value_counts()
# else:
#     bad_cusips_daily = pd.DataFrame()


all_fit_err = [x.fit_err for x in final_result[:-2]]
all_fit_err = np.hstack(all_fit_err)
quickplot(all_fit_err)


all_fit_err = [x.fit_err for x in final_result[:-2]]
all_fit_err = np.hstack(all_fit_err)
quickplot(all_fit_err)


# Drop some erroneous days
fit_err_spike = ((all_fit_err[1:-1] - all_fit_err[0:-2]) > .001) & ((all_fit_err[1:-1] - all_fit_err[2:]) > .001)
fit_err_spike = np.r_[False, fit_err_spike, False]
smooth_fit_err = all_fit_err[~fit_err_spike]
smooth_fit_err = np.convolve(np.r_[smooth_fit_err[0], smooth_fit_err, smooth_fit_err[-1]], np.ones((3,)) / 3.0, mode='valid')
quickplot(smooth_fit_err)

quickplot(fit_errs_stacked)

np.argmax(fit_errs_stacked)




# Look at individual dates
sel = 36
obs_date = obs_dates[sel] # If all were successful
obs_date = obs_dates_success[sel]



# Do some new fitting
EODMid = get_eod(obs_date)
importlib.reload(CRSPQuantlib)
res = CRSPQuantlib.FitSecuritiesQuantlib(EODMid.bond_maturities, EODMid.itype, EODMid.bond_rates, EODMid.bond_prices, EODMid.index, EODMid.bond_tdatdt, obs_date)
res.fit()


# Plot yield curve fit
quickplot(res.par_yields())
quickplot(res.yieldcurve_output[:,2])
quickplot(res.bond_yields_observed)

res.plot_par_curve_fit()


res.plot_fit_prices()
res.plot_fit_yields()

err_std = np.std(res.fit_err)
large_err = np.abs(res.fit_err) > (5 * err_std)
print("Absolute error: " + str(np.abs(res.fit_err[large_err])) + "      Relative error: " + str(np.abs(res.fit_err[large_err]) / err_std))


fig, axes = plt.subplots()
axes.plot(res.bond_plot_mats, res.bond_yields[:, 0], 'b.')
axes.plot(res.bond_plot_mats, res.bond_yields[:, 1], 'k.')


res.cusips[0]
GovPX_Today = GovPX[(GovPX.date == obs_date) & (GovPX.cusip == '912810CC')]
GovPX_Today

res.plot_fit_prices()


# Make plots of yield to maturity
opt = Quantlib_Helpers.QuantlibFitOptions()
obs_date_ql = Quantlib_Helpers.timestamp_to_qldate(obs_date)
bond_plot_mats = np.array([opt.day_counter.yearFraction(obs_date_ql, obs_date_ql + ql.Period(m, ql.Days))
                           for m in EODMid['bond_maturities']])
ytm = res[5][:, 0]


# Find some crazy stuff
fig, axes = plt.subplots()
plt.plot(EODMid.bond_prices)
EODMid.bond_prices.idxmax()
EODMid.bond_prices.idxmin()
EODMid.bond_prices.min()
# '9128274Y'  # random lowball quote in 1/1999???


fig, axes = plt.subplots()
plt.plot(bond_plot_mats, ytm * 100, 'b+')
plt.plot(res[4][:,0], par_yield * 100.0, color='red')
axes.set_xlim(0, 10.0)


fig, axes = plt.subplots()
plt.plot(res[5], 'o')




