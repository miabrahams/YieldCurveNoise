
import numpy as np
import os
from tqdm import tqdm
from mga_helpers import *
from CRSP_Helpers import *
import matplotlib.pyplot as plt
import QuantLib as ql
import CRSPQuantlib
import Quantlib_Helpers
import importlib
from lockfile import LockFile

importlib.reload(CRSPQuantlib)
importlib.reload(Quantlib_Helpers)

REFINE_ESTIMATES = True


# Load data.
CRSP_by_cusip = loadpklz('data/CRSP_by_cusip.pklz')
CRSP_by_cusip_useful = CRSP_by_cusip[['TMATDT', 'ITYPE', 'TCOUPRT', 'TDATDT']]

# You need to have run LoadGovPX already to get pklz.
GovPXFiles = [f for f in os.listdir('data\\GovPX') if f.endswith(".pklz")]
GovPXFiles = GovPXFiles[9:] + GovPXFiles[0:9] #Rearrange since 2000 is "GovPX00.zip"

# Gurkaynak, Sack and Wright starting values for parameters
gsw_params: pd.DataFrame = loadpklz("data/daily_gsw_params.pklz")



# TODO: Let's figure out what's going wrong with these hmm?  Also remember to exclude them later...
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
if not REFINE_ESTIMATES:
    final_result = []
    on_the_run_all = []
    # detailed_bond_info = []

GovPXFilesIter = [GovPXFiles[-1]]
GovPXFilesIter = GovPXFiles[0:8]
GovPXFilesIter = GovPXFiles


gsw_est_params = list(gsw_params.iloc[-1]) # Match GSW yields
yesterday_params = gsw_est_params # Track previous results
bounds_base = [np.array([  -3000,  -3000,  -3000,   -3000,   0,  0]),
               np.array([   3000,   2000,   3000,    3000,   20, 2])]
last_min_value = 0


if REFINE_ESTIMATES:
    outfile = 'data/temp/GovPX_fit_results_refined.pklz'
    lock = LockFile(outfile)
    lock.acquire()
    govpx_rs_final = loadpklz('data/temp/GovPX_fit_results_refined.pklz')
    lock.release()
    govpx_rs_final_dates = sorted(list(govpx_rs_final.keys()))
    govpx_rs_final_list = [govpx_rs_final[k] for k in govpx_rs_final_dates]
    # detailed_bond_info = []
    # detailed_bond_info_all = loadpklz('data/output/GovPX_daily_bond_info.pklz')
    improved_estimates = {}

for filename in GovPXFilesIter:
    print("____________ Loading ____________")
    print(filename)
    GovPX = loadpklz("data/GovPX/" + filename)
    obs_dates = GovPX.date.unique()
    detailed_bond_info = []

    ### Only estimate 30 days for debugging
    # obs_dates = obs_dates[1:30]
    obs_date = obs_dates[0]
    for obs_date in tqdm(obs_dates):
        EODInfo = get_eod(GovPX, obs_date)
        if EODInfo is None:
            continue
        # Load CRSP_prev_params here
        if REFINE_ESTIMATES:
            try:
                obs_index = int(np.where(govpx_rs_final_dates == obs_date)[0])
                previous_estimates = govpx_rs_final_list[obs_index]
                yesterday_params = govpx_rs_final_list[max(obs_index-1, 0)]['nss_params']
            except TypeError:
                print("Could not find prior estimates on " + nicedate(obs_date))
                continue
        try:
            gsw_est_params = list(gsw_params.loc[obs_date])
        except KeyError:
            print("GSW did not provide estimates on " + nicedate(obs_date))
            continue

        govpx_rs_ql = CRSPQuantlib.FitSecuritiesQuantlib(EODInfo.bond_maturities, EODInfo.itype, EODInfo.bond_rates, EODInfo.bond_prices, EODInfo.index, EODInfo.bond_tdatdt, obs_date)
        if REFINE_ESTIMATES:
            # parameter_start = [(p1 + p2) / 2.0 for p1, p2 in zip(previous_estimates['nss_params'], gsw_est_params)]
            parameter_start = govpx_rs_final_list[min(obs_index+1, len(govpx_rs_final_list))]['nss_params']
            parameter_start = previous_estimates['nss_params']
            # parameter_start = gsw_est_params
            if previous_estimates['end_criterion'] > 500:
                print("Going unbounded")
                bounds = None
            else:
                # Center bounds around last estimate
                param_sources = np.stack([bounds_base[0], gsw_est_params, yesterday_params, previous_estimates['nss_params']])
                # parameter_start = previous_estimates['nss_params']
                bounds_radius = (np.max(param_sources, 0) - np.min(param_sources, 0)) / 2
                bounds = [np.array(parameter_start) - bounds_radius, np.array(parameter_start) + bounds_radius]
        else:
            param_sources = np.stack([bounds_base[0], gsw_est_params, yesterday_params])
            parameter_start = gsw_est_params
            bounds =[np.min(param_sources, 0), np.max(param_sources, 0)]
            bounds_radius = (bounds[1] - bounds[0]) / 2

        try:
            # Start optim here
            govpx_rs_ql.check_problems = True
            if bounds is not None:
                bounds = [Quantlib_Helpers.ql_array(b) for b in bounds]
            govpx_fit = govpx_rs_ql.fit(parameter_start, bounds)
            params = govpx_rs_ql.nss_params_out
            if bounds is not None:
                hitBounds = [abs(p - low) < 1e-6 or abs(p  - high) < 1e-6 for (p, low, high) in zip(params, bounds[0], bounds[1])]
            if any(hitBounds):
                print("Hit parameter bounds on " + nicedate(obs_date))

                if REFINE_ESTIMATES:
                    # Recenter bounds to lie around the new start point
                    bounds = [np.array(params) - bounds_radius, np.array(params) + bounds_radius]
                    # We'll save this to disk to make sure we record the constraint issue
                    previous_estimates['end_criterion'] = 50
                    improved_estimates[obs_date] = previous_estimates
                    govpx_fit = govpx_rs_ql.fit(params, bounds)
                    govpx_fit['end_criterion'] = 50
                    if any([abs(p - low) < 1e-6 or abs(p  - high) < 1e-6 for (p, low, high) in zip(govpx_rs_ql.nss_params_out, bounds[0], bounds[1])]):
                        print("Hit bounds twice!")
                        govpx_fit['end_criterion'] = 100
                else:
                    govpx_fit = govpx_rs_ql.fit(params, bounds)
                    govpx_fit['end_criterion'] = 50
                    bounds = None # Unbounded

            # if (govpx_fit['min_value'] - last_min_value) > .0001:
            if bounds is not None:
                bounds = [np.array(yesterday_params) - bounds_radius, np.array(yesterday_params) + bounds_radius]
            govpx_fit2 = govpx_rs_ql.fit(yesterday_params, bounds)
            if govpx_fit2['min_value'] < govpx_fit['min_value']:
                govpx_fit = govpx_fit2
        except RuntimeError as e:
            print("Failed to fit on " + nicedate(obs_date) + ". Reason: " + str(e))
            continue

        ## Diagnostics
        # gsw_yield_today = Quantlib_Helpers.nss_yield(gsw_est_params)
        # est_yield_today = Quantlib_Helpers.nss_yield(last_params)
        # pd.DataFrame({"GSW": gsw_yield_today, "Est": est_yield_today}).plot()
        # print(last_params)
        # print(gsw_est_params)
        # print(null_params)
        # res_final.plot_fit_yields()  # For diagnostics
        ## End diagnostics

        # Detailed bond info by cusip
        # TODO: correlate time with stuff
        EODInfo_by_cusip = EODInfo.loc[govpx_rs_ql.cusips]
        EOD_Time = EODInfo_by_cusip.time.astype('int64')
        govpx_fit['mean_time'] = pd.to_datetime(EOD_Time.mean().astype('datetime64[ns]'))
        govpx_fit['std_time'] = EOD_Time.std().astype('timedelta64[ns]').astype('timedelta64[m]')
        on_the_run_cusips = EODInfo.active.loc[EODInfo.active == 'A'].index.tolist()
        def _cusipStatus(c):
            if c in on_the_run_cusips:
                return 1
            elif c in govpx_rs_ql.bad_cusips:
                return -1
            return 0
        cusipStatus = np.array([_cusipStatus(c) for c in govpx_rs_ql.cusips])
        daily_bond_info = {"cusip": govpx_fit['cusips'], "date": np.repeat(obs_date, govpx_fit['cusips'].shape),
                           "price": govpx_fit['bond_prices'], "yield": govpx_fit['bond_yields'],
                           "status": cusipStatus, "spread": EODInfo_by_cusip.spread, "time": EODInfo_by_cusip.time,
                           "bad_bonds": govpx_rs_ql.bad_bond_info}

        detailed_bond_info.append(daily_bond_info)
        if REFINE_ESTIMATES:
            # detailed_bond_info_all[obs_date] = daily_bond_info
            if govpx_fit['min_value'] < previous_estimates['min_value']:
                print("Improved at date " + nicedate(obs_date) + "from " +
                      str(previous_estimates['min_value']) + " to " + str(govpx_fit['min_value']))
                # govpx_rs_final[obs_date] = govpx_fit
                improved_estimates[obs_date] = govpx_fit
                last_min_value = govpx_fit['min_value']
            else:
                last_min_value = previous_estimates['min_value']
                previous_estimates['min_value'] = govpx_fit['min_value']
                improved_estimates[obs_date] = previous_estimates
                # print("No improvement at date " + nicedate(obs_date))
        else:
            final_result.append(govpx_fit)
            # detailed_bond_info += daily_bond_info
            on_the_run_all.append(pd.DataFrame({"date": np.repeat(obs_date, len(on_the_run_cusips)),
                                                "price": EODInfo.bond_prices.loc[EODInfo.active == 'A'].tolist(),
                                                "cusip": on_the_run_cusips}))

    yearnum = filename.split('.')[0][-2:]
    if not REFINE_ESTIMATES:
        final_result_out = {r['obs_date']: r for r in final_result}
        savepklz(final_result_out, "data/temp/GovPX_Daily_Result" + yearnum + ".pklz")
        savepklz(detailed_bond_info, "data/temp/GovPX_daily_bond_info" + yearnum + ".pklz")
        savepklz(pd.concat(on_the_run_all), "data/temp/GovPX_on_the_run_cusips" + yearnum + ".pklz")
    else:
        print("SAVING -")
        outfile = 'data/temp/GovPX_fit_results_refined.pklz'
        lock = LockFile(outfile)
        lock.acquire()
        up_to_date_results = loadpklz(outfile)
        for date, est in improved_estimates.items():
            up_to_date_results[date] = est
        savepklz(up_to_date_results, outfile)
        improved_estimates = {}
        lock.release()



IS_SCRIPT = True

if not IS_SCRIPT:

    # if PARTIAL and not FIRST:
    #     savepklz(pd.concat(final_result), "data/temp/GovPX_Daily_Result.pklz")
    #     savepklz(pd.concat(on_the_run_all), "data/temp/on_the_run_cusips.pklz")
    #     savepklz(pd.DataFrame({'GovPXFile': GovPXFiles, 'n_dropped': n_days_dropped}), "data/temp/n_days_dropped.pklz")
    #     savepklz(detailed_bond_info, "data/temp/GovPX_daily_bond_info.pklz")
    #

    govpx_rs_final = {}
    detailed_bond_info_all = {}
    for filename in GovPXFiles:
        yearnum = filename.split('.')[0][-2:]
        rs = loadpklz('data/temp/GovPX_daily_Result' + yearnum + '.pklz')
        for date, value in rs.items():
            govpx_rs_final[date] = value
        # detailed_bond_info = loadpklz("data/temp/GovPX_daily_bond_info" + yearnum + ".pklz")
        # for est in detailed_bond_info:
        #     detailed_bond_info_all[est.date[0]] = est

    savepklz(govpx_rs_final, 'data/output/GovPX_fit_results.pklz')
    savepklz(govpx_rs_final, 'data/temp/GovPX_fit_results_refined.pklz')





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

    fit_errs_stacked = []
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




