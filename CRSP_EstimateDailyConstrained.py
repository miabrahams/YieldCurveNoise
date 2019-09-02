
import QuantLib as ql
from tqdm import tqdm
import importlib
import CRSPQuantlib
from CRSP_Helpers import *
from mga_helpers import *
from Quantlib_Helpers import *
import numpy as np
importlib.reload(CRSPQuantlib)

from lockfile import LockFile

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

# Set to some number.
batch = 55
batches = 150
included_batches = range(batches) # In case we don't want to do all of them
# included_batches = range(20) # drop 20?

print("CONSTRAINED ESTIMATES - NO PENALTY")

DO_FINALIZE = False
DO_EQUAL_WEIGHT = True
REFINE_ESTIMATES = True

FIT_METHOD = CRSPQuantlib.FitMethod.NELDER_MEAD
# FIT_METHOD = CRSPQuantlib.FitMethod.ANNEALING
# FIT_METHOD = CRSPQuantlib.FitMethod.DE

batch_file_path = 'data/Constrained/CRSP_fit_result_'
destfile = 'data/Constrained/CRSP_fit_results.pklz'
refinefile = 'data/Constrained/CRSP_fit_results_refined.pklz'
refinefile_list = 'data/Constrained/CRSP_fit_results_list_refined.pklz'


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
print ("Estimation starting on " + nicedate(selected_dates[0]))

bounds_base = [np.array([      0,    -15,  -30,   -30,   1.0/2.5,  1/5.5]),
               np.array([     15,     30,   30,    30,   200, 1/2.5])]
bounds_hard = [ql_array(b) for b in bounds_base]


def clip_at_bounds(p):
    return np.clip(np.array(p), bounds_base[0], bounds_base[1])

if not DO_FINALIZE:
    # Note: should replace this with NSS params
    if REFINE_ESTIMATES:
        print(" REFINING BATCH " + str(batch))
        lock = LockFile(refinefile)
        lock.acquire()
        rs_final = loadpklz(refinefile)
        lock.release()
        rs_final_dates = sorted(list(rs_final.keys()))
        rs_final_list = [rs_final[k] for k in rs_final_dates]
        improved_estimates = {}
    else:
        print("STARTING BATCH " + str(batch))
        rs = []

    gsw_est_params = clip_at_bounds(gsw_params.iloc[-1])
    yesterday_params = np.mean(bounds_base, 0)
    last_min_value = 0
    obs_date = selected_dates[0]
    for obs_date in tqdm(selected_dates):
        npdata2 = npdata[cal_date == obs_date,:]
        # Load CRSP_prev_params here
        if REFINE_ESTIMATES:
            try:
                obs_index = int(np.where(rs_final_dates == obs_date)[0])
                previous_estimates = rs_final_list[obs_index]
                previous_estimates['nss_params'] = clip_at_bounds(previous_estimates['nss_params'])
                # yesterday_params = .5 * yesterday_params + .5 * np.array(rs_final_list[max(obs_index-1, 0)]['nss_params'])
                yesterday_params = .5 * yesterday_params + .5 * previous_estimates['nss_params']
            except TypeError:
                print("Estimation previously failed on " + nicedate(obs_date))
                # Keep yesterday_params and use gsw params from day before
                previous_estimates['nss_params'] = gsw_est_params
                previous_estimates['min_value'] = 10e10
                # continue
        try:
            gsw_est_params = clip_at_bounds(np.array(gsw_params.loc[obs_date]))
        except KeyError:
            print("GSW did not estimate the yield curve on " + nicedate(obs_date))
        try:
            rs_ql = CRSPQuantlib.FitSecuritiesQuantlib(npdata2[:, 0], npdata2[:, 1], npdata2[:, 2], npdata2[:, 3], npdata2[:,4], bond_tdatdt[cal_date == obs_date], obs_date)
            rs_ql.equal_weight = DO_EQUAL_WEIGHT
            rs_ql.fit_method = FIT_METHOD
            # Find appropriate bounds based on parameter estimates
            if REFINE_ESTIMATES:
                # param_sources = np.stack([bounds_base[0], gsw_est_params, yesterday_params, previous_estimates['nss_params']])
                # parameter_start = rs_final_list[min(obs_index+1, len(unique_dates_np))]['nss_params']
                # parameter_start = rs_final_list[max(obs_index-1, 0)]['nss_params']
                parameter_start = previous_estimates['nss_params']
            else:
                # param_sources = np.stack([bounds_base[0], gsw_est_params, yesterday_params])
                parameter_start = gsw_est_params
                # bounds =[np.min(param_sources, 0), np.max(param_sources, 0)]
                # bounds_radius = (bounds[1] - bounds[0]) / 2

            # Start optim here
            fit = rs_ql.fit(ql_array(parameter_start), bounds_hard)
            # print(rs_ql.yieldcurve.fitResults().weights())

            # # Assign other values of params?
            # params = rs_ql.nss_params_out
            # params_tmp = params
            # costs = []
            # for l in tqdm(np.arange(0, 2.5, 0.02)):
            #     params_tmp[4] = l
            #     rs_ql.assign_nss_params(params_tmp)
            #     costs.append(rs_ql.yieldcurve.fitResults().minimumCostValue())

            # If we hit bounds, try again
            # params = rs_ql.nss_params_out
            # if bounds is not None:
            #     hitBounds = [abs(p - low) < 1e-6 or abs(p  - high) < 1e-6 for (p, low, high) in zip(params, bounds[0], bounds[1])]
            # elif any(hitBounds):
            #     print("Hit parameter bounds on " + nicedate(obs_date))
            #     if REFINE_ESTIMATES:
            #         # Recenter bounds to lie around the new start point
            #         bounds = [np.array(params) - bounds_radius, np.array(params) + bounds_radius]
            #         # We'll save this to disk to make sure we record the constraint issue
            #         previous_estimates['end_criterion'] = 50
            #         improved_estimates[obs_date] = previous_estimates
            #         fit = rs_ql.fit(params, bounds)
            #         fit['end_criterion'] = 50
            #         if any([abs(p - low) < 1e-6 or abs(p  - high) < 1.0e-6 for (p, low, high) in zip(rs_ql.nss_params_out, bounds[0], bounds[1])]):
            #             print("Hit bounds twice!")
            #             fit['end_criterion'] = 100
            #     else:
            #         fit = rs_ql.fit(params, bounds)
            #         bounds = None # Unbounded

            # if (fit['min_value'] - last_min_value) > .0001:
                # if bounds is not None:
                #     bounds = [np.array(yesterday_params) - bounds_radius, np.array(yesterday_params) + bounds_radius]
            if FIT_METHOD is CRSPQuantlib.FitMethod.NELDER_MEAD:
                # Other methods are too slow
                fit2 = rs_ql.fit(yesterday_params, bounds_hard)
                if (fit['min_value'] - fit2['min_value']) > 10e-8:
                    # print("Improved fit by searching near yesterday's params. Moved from " + str(fit['min_value']) + " to " + str(fit2['min_value']))
                    fit = fit2

            if REFINE_ESTIMATES:
                pct_improvement = 1 - (fit['min_value'] / previous_estimates['min_value'])
                if pct_improvement > 0:
                    improved_estimates[obs_date] = fit
                    if pct_improvement > .01:
                        print(" Improved fit on {} by {:.2%} from {: f} to {: f}".format(
                            nicedate(obs_date), pct_improvement,
                            previous_estimates['min_value'], fit['min_value']))
                    # rs_final[obs_date] = fit
                else:
                    print("No improvement - best result {:.2%} ".format(pct_improvement) + nicedate(obs_date))
                    previous_estimates['min_value'] = fit['min_value']
            else:
                rs.append(fit)
                last_params = rs_ql.nss_params_out

            last_min_value = fit['min_value']
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
    else:
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


