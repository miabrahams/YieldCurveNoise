
import QuantLib as ql
from tqdm import tqdm
import importlib
import CRSPQuantlib
from CRSP_Helpers import *
from mga_helpers import *
from Quantlib_Helpers import *
import numpy as np
import sys

importlib.reload(CRSPQuantlib)

from lockfile import LockFile

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)




# Set to some number.
batch = 1
batches = 10
included_batches = range(batches) # In case we don't want to do all of them
# included_batches = range(20) # drop 20?

DO_FINALIZE = False
DO_EQUAL_WEIGHT = False
REFINE_ESTIMATES = True
DO_RIDGE = True
WRITE_LIST = False

if len(sys.argv) > 2:
    print(" Got argv~ " + str(sys.argv))


FIT_METHOD = CRSPQuantlib.FitMethod.NELDER_MEAD
# FIT_METHOD = CRSPQuantlib.FitMethod.ANNEALING
# FIT_METHOD = CRSPQuantlib.FitMethod.DE


if DO_EQUAL_WEIGHT:
    batch_file_path = 'data/RegularizedPrice/CRSP_fit_result_'
    destfile = 'data/RegularizedPrice/CRSP_fit_results.pklz'
    refinefile = 'data/RegularizedPrice/CRSP_fit_results_refined.pklz'
    refinefile_list = 'data/RegularizedPrice/CRSP_fit_results_list_refined.pklz'
else:
    batch_file_path = 'data/Regularized/CRSP_fit_result_'
    destfile = 'data/Regularized/CRSP_fit_results.pklz'
    refinefile = 'data/Regularized/CRSP_fit_results_refined.pklz'
    refinefile_list = 'data/Regularized/CRSP_fit_results_list_refined.pklz'

ql.EndCriteria




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


# Select the batch to run
selection = chunk(len(unique_dates), batches)[batch - 1]
selected_dates = unique_dates_np[selection]
print("Estimation of batch {} starting on {}".format(batch, nicedate(selected_dates[0])))
if DO_EQUAL_WEIGHT:
    print("Fitting price error.")
if REFINE_ESTIMATES:
    print("Refining previous estimates.")
print("Optimization Method: " + str(FIT_METHOD))

bounds_base = [np.array([      0,    -15,  -30,   -30,   1.0/2.5,  1/5.5]),
               np.array([     15,     30,   30,    30,   200, 1/2.5])]
bounds_hard = [ql_array(b) for b in bounds_base]

np.random.uniform(bounds_base[0], bounds_base[1] - np.array([5, 0, 0, 0, 0, 0]))


def clip_at_bounds(p):
    return np.clip(np.array(p), bounds_base[0], bounds_base[1])

if not DO_FINALIZE:
    # Note: should replace this with NSS params
    if REFINE_ESTIMATES:
        lock = LockFile(refinefile)
        lock.acquire()
        rs_final = loadpklz(refinefile)
        lock.release()
        rs_final_dates = sorted(list(rs_final.keys()))
        rs_final_list = [rs_final[k] for k in rs_final_dates]
        improved_estimates = {}
        if selection.start == 0:
            yesterday_params = np.array(rs_final_list[1]['nss_params'])
            two_day_ago_params = np.array(rs_final_list[96]['nss_params'])
            rolling_avg_params = yesterday_params
            print("For first observation, setting prior parameters to " + str(yesterday_params))
            print("Current estimate was " + str(rs_final_list[0]['nss_params']))
        else:
            yesterday_params = np.array(rs_final_list[selection.start - 1]['nss_params'])
            two_day_ago_params = np.array(rs_final_list[selection.start - 2]['nss_params'])
            rolling_avg_params = yesterday_params
        parameter_start2 = yesterday_params
    else:
        simple_prior = np.array([5.0, 1.0, 1.0, 1.0, 1/1.37, 1/3])
        yesterday_params = np.array([5.0, 1.0, 1.0, 1.0, 1/1.37, 1/3])
        two_day_ago_params = yesterday_params
        rs = []

    # last_min_value = 0
    gsw_est_params = clip_at_bounds(gsw_params.iloc[-1])
    obs_date = selected_dates[0]
    for obs_date in tqdm(selected_dates):
        npdata2 = npdata[cal_date == obs_date,:]
        # Load CRSP_prev_params here
        if REFINE_ESTIMATES:
            try:
                obs_index = int(np.where(rs_final_dates == obs_date)[0])
                previous_estimates = rs_final_list[obs_index]
            except TypeError:
                print("Estimation previously failed on " + nicedate(obs_date))
                # Keep yesterday_params and use gsw params from day before
                previous_estimates['nss_params'] = gsw_est_params
                previous_estimates['min_value'] = 10e10
                # continue
        try:
            rs_ql = CRSPQuantlib.FitSecuritiesQuantlib(npdata2[:, 0], npdata2[:, 1], npdata2[:, 2], npdata2[:, 3], npdata2[:,4], bond_tdatdt[cal_date == obs_date], obs_date)
            rs_ql.equal_weight = DO_EQUAL_WEIGHT
            rs_ql.fit_method = FIT_METHOD
            # Find appropriate bounds based on parameter estimates
            if REFINE_ESTIMATES:
                param_sources = np.stack([bounds_base[0], gsw_est_params, yesterday_params, previous_estimates['nss_params']])
                # parameter_start = rs_final_list[min(obs_index+1, len(unique_dates_np))]['nss_params']
                # parameter_start = rs_final_list[max(obs_index-1, 0)]['nss_params']
                # tomorrow_params = np.array(rs_final_list[min(obs_index+1, len(rs_final_list) - 1)]['nss_params'])
                # parameter_start1 = previous_estimates['nss_params']
                parameter_start1 = rolling_avg_params
                parameter_start2 = yesterday_params
                # parameter_start2 = (yesterday_params + tomorrow_params) / 2.0
                # Use an exponential decay and lookahead
                # parameter_start2 = .25 * parameter_start2 + .75 * (yesterday_params)
                # Randomize
                # random_draw = np.random.uniform(bounds_base[0], bounds_base[1] - np.array([5, 0, 0, 0, 0, 0]))
                # parameter_start2 = .1 * random_draw + .9 / 2.0 * (yesterday_params + parameter_start1)
                # rs_ql.regularize_prior = (yesterday_params + two_day_ago_params) / 2.0
                # rs_ql.regularize_prior = (yesterday_params + tomorrow_params) / 0.5
                rs_ql.regularize_prior = yesterday_params
            else:
                try:
                    gsw_est_params = clip_at_bounds(np.array(gsw_params.loc[obs_date]))
                except KeyError:
                    print("GSW did not estimate the yield curve on " + nicedate(obs_date))
                parameter_start1 = gsw_est_params
                parameter_start2 = yesterday_params
                rs_ql.regularize_prior = yesterday_params

            # two_day_ago_params = yesterday_params
            rolling_avg_params = .8 * rolling_avg_params + .2 * yesterday_params

            # Start optim here
            if FIT_METHOD is CRSPQuantlib.FitMethod.NELDER_MEAD:
                fit = rs_ql.fit_quickly(ql_array(parameter_start1), bounds_hard)
                fit2 = rs_ql.fit_quickly(parameter_start2, bounds_hard)
                if (fit['min_value'] - fit2['min_value']) > 10e-8:
                    # Use new fit
                    fit = rs_ql.fit_quickly_finalize()
                    # print("Randomization helped!  {: f}  vs  {: f}".format(fit2['min_value'], fit['min_value']))
                    # print("Lookahead helped!  {: f}  vs  {: f}".format(fit2['min_value'], fit['min_value']))
                else:
                    rs_ql.assign_nss_params(fit['nss_params'])
                    fit = rs_ql.fit_quickly_finalize()
                    fit['min_value'] = fit2['min_value']
                    fit['end_criterion'] = fit2['end_criterion']
            else:
                fit = rs_ql.fit(ql_array(parameter_start1), bounds_hard)

            if REFINE_ESTIMATES:
                pct_improvement = 1 - (fit['min_value'] / previous_estimates['min_value'])

                # Clean up spikes
                if np.isin(obs_date, spike_dates):
                    saved_yesterday_params = yesterday_params
                    print("______________")
                    print("HIT A SPIKE DAY")
                    print("______________")
                else:
                    saved_yesterday_params = None

                print("")
                print(" Termination condition: " + str(fit['end_criterion']))
                print(" n_iter: " + str(rs_ql.yieldcurve.fitResults().numberOfIterations()))
                print(" New parameter values: " + str(np.array(fit['nss_params'])))
                print(" Old parameter values: " + str(np.array(previous_estimates['nss_params'])))
                print(" Contribution from regularization: " +
                      str(sum(np.square(np.array(fit['nss_params']) - yesterday_params)) * .01))
                if pct_improvement > 0:
                    improved_estimates[obs_date] = fit
                    yesterday_params = np.array(fit['nss_params'])
                    if pct_improvement > .01:
                        print(" Improved fit on {} by {:.2%} from {: f} to {: f}".format(
                            nicedate(obs_date), pct_improvement,
                            previous_estimates['min_value'], fit['min_value']))
                    # rs_final[obs_date] = fit
                elif DO_RIDGE:  # We don't get to hold over the value function...
                    improved_estimates[obs_date] = fit
                    yesterday_params = np.array(fit['nss_params'])
                else:
                    yesterday_params = np.array(previous_estimates['nss_params'])
                #     previous_estimates['min_value'] = fit['min_value']
                #     improved_estimates[obs_date] = previous_estimates
                #     print("No improvement at date " + nicedate(obs_date))
                # print(" Parameters updated: " + str(np.array(fit['nss_params']) - np.array(previous_estimates['nss_params'])))
                if saved_yesterday_params:
                    yesterday_params = saved_yesterday_params
            else:
                rs.append(fit)
                last_params = rs_ql.nss_params_out
                yesterday_params = np.array(fit['nss_params'])
                print("Final value was {}".format(fit['min_value']))


            # last_min_value = fit['min_value']
        except Exception as e:
            print("Calculation on date " + nicedate(obs_date) + " failed! Exception: " + str(e))
            # if REFINE_ESTIMATES:
            #     break

    if REFINE_ESTIMATES:
        print("SAVING - ")
        lock = LockFile(refinefile)
        lock.acquire()
        up_to_date_results = loadpklz(refinefile)
        for date, est in improved_estimates.items():
            up_to_date_results[date] = est
        savepklz(up_to_date_results, refinefile)
        lock.release()

        # if WRITE_LIST:
        #     crsp_results_list = []
        #     for date in tqdm(unique_dates_np):
        #         try:
        #             crsp_results_list.append(up_to_date_results[date])
        #         except KeyError as e:
        #             continue
        #     savepklz((crsp_results_list, refinefile_list), )

    else:
        # from matplotlib import pyplot as plt
        # params_matrix = np.array([fit['nss_params'] for fit in rs])
        # plt.plot(np.array([fit['obs_date'] for fit in rs]), params_matrix[:,0] + params_matrix[:,1])
        # plt.plot(np.array([fit['obs_date'] for fit in rs]), params_matrix[:,0])
        # plt.plot(np.array([fit['obs_date'] for fit in rs]), params_matrix[:,2])
        # plt.plot(np.array([fit['obs_date'] for fit in rs]), params_matrix[:,3])
        # plt.plot(np.array([fit['obs_date'] for fit in rs]), params_matrix[:,4])
        # plt.plot(np.array([fit['obs_date'] for fit in rs]), params_matrix[:,5])
        savepklz(rs, batch_file_path + str(batch) + '.pklz')

## Not doing estimation
really_finalize = False
if really_finalize:
    # Script for combining multiple files together
    rs_final = {}
    for batch in included_batches:
       try:
           rs = loadpklz(batch_file_path + str(batch + 1) + '.pklz')
           for est in rs:
               rs_final[est['obs_date']] = est
       except Exception as e:
           print(e)
           continue

    savepklz(rs_final, destfile)


if DO_FINALIZE:
    print("Writing list:")
    CRSPResults = loadpklz(refinefile)
    CRSPdates = sorted(list(CRSPResults.keys()))
    CRSPResults_list = [CRSPResults[d] for d in CRSPdates]
    savepklz((CRSPdates, CRSPResults_list), refinefile_list)
    print("Done!")





# Some plotting functions
do_plots = False
if do_plots:
    lock = LockFile(refinefile)
    lock.acquire()
    rs_final = loadpklz(refinefile)
    print("Hello")
    rs_final_list = []
    rs_final_dates = []
    for d in unique_dates_np:
        # print(nicedate(d))
        try:
            rs_final_list.append(rs_final[d])
            rs_final_dates.append(d)
        except KeyError:
            print("Could not find date " + nicedate(d))
    del rs_final
    lock.release()

    fit_err = [r['fit_err'] for r in rs_final_list]
    price_err = [r['price_err'] for r in rs_final_list]
    min_value = [r['min_value'] for r in rs_final_list]
    fit_err_plot = pd.DataFrame({'fit_err': fit_err, 'price_err': price_err}, index=np.array(rs_final_dates))
    # fit_err_plot.fit_err.plot()

    spike_dates = pd.Series(pd.to_datetime(['1996-04-05', '1998-01-23', '1998-07-10', '1998-07-21', '1998-10-07', '1999-01-28', '2006-06-30', '2011-08-01', '2011-08-03', '2011-08-05'])).as_matrix()
    fit_err_plot = fit_err_plot.loc[fit_err_plot.index.drop(spike_dates)]


    ax = fit_err_plot.fit_err.plot(style='k')
    ax.get_lines()[0].set_linewidth(.5)


    ax = fit_err_plot.fit_err.resample('1W').mean().plot(style='k')
    ax.get_lines()[0].set_linewidth(.5)


    ax = (fit_err_plot.price_err.apply(np.log).resample('1W').mean() / 100).plot(style='k')
    ax.get_lines()[0].set_linewidth(.5)

    CRSPdates = np.array([r['obs_date'] for r in rs])
    CRSP_fit_err_std_raw = np.array([r['fit_err'] for r in rs])
    CRSP_fit_err_std = pd.Series(CRSP_fit_err_std_raw, index=CRSPdates)
    CRSP_fit_err_std.plot()


