
from tqdm import tqdm
import importlib
import CRSPQuantlib
from CRSP_Helpers import *
from mga_helpers import *
from Quantlib_Helpers import *
import numpy as np
importlib.reload(CRSPQuantlib)
from lockfile import LockFile


# Support command line input
args = None
if __name__ == "__main__":
    import __main__ as main
    if hasattr(main, '__file__'):
        import argparse
        parser = argparse.ArgumentParser(description='Specify estimation')
        parser.add_argument('-b', metavar='Batch', type=int, nargs=2, help="Batch number and total batch count")
        parser.add_argument('-ew', metavar='EqualWeight', type=str2bool, nargs=1, help="Do equal weighting")
        parser.add_argument('-zr', metavar='ZeroRidge', type=str2bool, nargs=1, help="Ridge regression")
        args = parser.parse_args()


np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

# Set to some number.
batch = 1
batches = 10
included_batches = range(batches) # In case we don't want to do all of them
# included_batches = range(20) # drop 20?



DO_FINALIZE = False
REFINE_ESTIMATES = True
DO_EQUAL_WEIGHT = False
DO_ZERO_RIDGE = False
# We're going to use median instead atm
# MIN_CUTOFF = 100 # Cutoff for likelihood value, set to 0 to estimate every day
NUM_ITER = 1

FIT_METHOD = CRSPQuantlib.FitMethod.NELDER_MEAD
# FIT_METHOD = CRSPQuantlib.FitMethod.ANNEALING
# FIT_METHOD = CRSPQuantlib.FitMethod.DE


if args is not None:
    print(args)
    if args.b is not None:
        batch = args.b[0]
        batches = args.b[1]
    if args.ew is not None:
        DO_EQUAL_WEIGHT = args.ew[0]
    if args.zr is not None:
        DO_ZERO_RIDGE = args.zr[0]

print("__DOING COMPARISON__")
comparefile = 'data/ZeroRidge/CRSP_fit_results_list_refined.pklz'

if DO_ZERO_RIDGE:
    if DO_EQUAL_WEIGHT:
        batch_file_path = 'data/EWZeroRidge/CRSP_fit_result_'
        destfile = 'data/EWZeroRidge/CRSP_fit_results.pklz'
        refinefile = 'data/EWZeroRidge/CRSP_fit_results_refined.pklz'
        refinefile_list = 'data/EWZeroRidge/CRSP_fit_results_list_refined.pklz'
    else:
        batch_file_path = 'data/ZeroRidge/CRSP_fit_result_'
        destfile = 'data/ZeroRidge/CRSP_fit_results.pklz'
        refinefile = 'data/ZeroRidge/CRSP_fit_results_refined.pklz'
        refinefile_list = 'data/ZeroRidge/CRSP_fit_results_list_refined.pklz'
else:
    if DO_EQUAL_WEIGHT:
        batch_file_path = 'data/EqualWeight/CRSP_fit_result_'
        destfile = 'data/EqualWeight/CRSP_fit_results.pklz'
        refinefile = 'data/EqualWeight/CRSP_fit_results_refined.pklz'
        refinefile_list = 'data/EqualWeight/CRSP_fit_results_list_refined.pklz'
    else:
        batch_file_path = 'data/temp/CRSP_fit_result_'
        destfile = 'data/output/CRSP_fit_results.pklz'
        refinefile = 'data/temp/CRSP_fit_results_refined.pklz'
        refinefile_list = 'data/temp/CRSP_fit_results_list_refined.pklz'



CRSPData = pd.read_pickle("data/CRSP_Treasuries.pkl.compress", compression="gzip")
CRSPData = clean_bad_bonds(CRSPData)
unique_dates_raw = pd.read_pickle("data/CRSPUniqueDates.pkl.compress", compression="gzip")
unique_dates_np = pd.to_datetime(pd.Series([u for u in unique_dates_raw if u not in bad_dates])).as_matrix()
del unique_dates_raw

# Select the batch to run
selection = chunk(len(unique_dates_np), batches)[batch - 1]
selected_dates = unique_dates_np[selection]

CRSPData = CRSPData[(CRSPData.CALDT < selected_dates[-1]) & (CRSPData.CALDT >= selected_dates[0])]
# CRSPData = CRSPData[np.all((CRSPData.CALDT < selected_dates[-1]), (CRSPData.CALDT >= selected_dates[0]))]



# Gurkaynak, Sack and Wright starting values for parameters
gsw_params: pd.DataFrame = loadpklz("data/daily_gsw_params.pklz")


# Get everything ready for fast parameter passing
bond_maturities = CRSPData.ttm.as_matrix()
itype = CRSPData.ITYPE.as_matrix()
caldt = CRSPData.CALDT.as_matrix()
bond_rates = CRSPData.TCOUPRT.as_matrix()
bond_prices = CRSPData.TDNOMPRC.as_matrix()
cusips = CRSPData.TCUSIP.as_matrix()
cal_date = CRSPData.CALDT.astype('category')
bond_tdatdt = CRSPData.TDATDT.as_matrix()


# from pandas.api.types import CategoricalDtype
# cal_date = CRSPData.CALDT.astype(CategoricalDtype(ordered=True, categories= selected_dates))


# Stack everything into a single ndarray for speed. Stacking datetime64 together with other types doesn't work.
npdata = np.stack([bond_maturities, itype, bond_rates, bond_prices, cusips], axis=1)

# Clean the namespace
del CRSPData, bond_maturities, itype, caldt, bond_rates, bond_prices, cusips, unique_dates_np




# Some constants used in the iterations
zeroPrior   =  np.array([0, 0, 0, 0, 0, 0])
bounds_base = [np.array([      0, -20,  -200,   -200,   .4,  0]),
               np.array([   15,  200,   200,    200,   60, .4])]

bounds_mean = np.array([2.5, 2.5, 10, 10, 1, .2])

bounds_crazy = [np.array([      0, -9000, -9000,  -9000,      0,    0]),
                np.array([   9000,  9000,  9000,   9000,   5000, 5000])]

if DO_FINALIZE:
    print("Constructing results list")
else:
    print(f"Estimation of batch {batch}/{batches} starting on {nicedate(selected_dates[0])}")
    if REFINE_ESTIMATES:
        print("Refining previous estimates.")
    print("Optimization Method: " + str(FIT_METHOD))
if DO_EQUAL_WEIGHT:
    print("Equal weighting.")
if DO_ZERO_RIDGE:
    print("Ridge regression.")

def clip_at_bounds(p):
    return np.clip(np.array(p), bounds_base[0], bounds_base[1])
def clip_at_outer_bounds(p):
    return np.clip(np.array(p), bounds_crazy[0], bounds_crazy[1])
def add_rand(p):
    return np.random.normal(p, np.array([1, 1, 10, 20, .2, .05]))

if not DO_FINALIZE:
    while NUM_ITER > 0:
        NUM_ITER -= 1
        # Note: should replace this with NSS params
        if REFINE_ESTIMATES:
            lock = LockFile(refinefile)
            lock.acquire()
            rs_final = loadpklz(refinefile)
            lock.release()
            rs_final_dates = sorted(list(rs_final.keys()))
            rs_final_list = [rs_final[k] for k in rs_final_dates]
            improved_estimates = {}
            del rs_final

            # Note: assuming we get the same set of dates in the same order here
            (_, rs_compare_list) = loadpklz(comparefile)
            rs_compare_params = [r['nss_params'] for r in rs_compare_list]
            del _, rs_compare_list

            MEDIAN_OPTIM_VAL = np.median([r['min_value'] for r in rs_final_list])
            print(f"Skipping estimates with value less than {MEDIAN_OPTIM_VAL * .5}")
        else:
            print("STARTING BATCH " + str(batch))
            rs = []

        gsw_est_params = list(gsw_params.iloc[-1])
        yesterday_params = list(gsw_est_params)
        yesterday_min_value = 0
        for obs_date in tqdm(selected_dates):
            # For debugging
            # obs_date = selected_dates[206]
            # Load CRSP_prev_params here
            if REFINE_ESTIMATES:
                try:
                    obs_index = int(np.where(rs_final_dates == obs_date)[0])
                    previous_estimates = rs_final_list[obs_index]
                    compare_estimates = rs_compare_params[obs_index]
                    previous_min = previous_estimates['min_value'] + .001
                    previous_nss = previous_estimates['nss_params']
                    yesterday_params = rs_final_list[max(obs_index-1, 0)]['nss_params']
                    # if not np.all(clip_at_outer_bounds(previous_nss) == np.array(previous_nss)):
                    #     tqdm.write(f"Previous estimates on {nicedate(obs_date)} were far out of bounds: {previous_nss}")
                    #     previous_min = 10e10
                except Exception as e:
                    tqdm.write("Estimation previously failed on " + nicedate(obs_date))
                    # Keep yesterday_params and use gsw params from day before
                    previous_nss = gsw_est_params
                    previous_min = 10e10
                    # continue
            try:
                gsw_est_params = list(gsw_params.loc[obs_date])
            except KeyError:
                tqdm.write("GSW did not estimate the yield curve on " + nicedate(obs_date))
            try:
                npdata_current = npdata[cal_date == obs_date,0:5]
                rs_ql = CRSPQuantlib.FitSecuritiesQuantlib(*[npdata_current[:,c] for c in range(0,5)], bond_tdatdt[cal_date == obs_date], obs_date)
                del npdata_current

                rs_ql.equal_weight = DO_EQUAL_WEIGHT
                rs_ql.fit_method = FIT_METHOD
                # Find appropriate bounds based on parameter estimates
                if REFINE_ESTIMATES:
                    if (previous_min < (MEDIAN_OPTIM_VAL*0)) and (yesterday_min_value > previous_min):
                            # Only if yesterday AND tomorrow are both higher!
                            tomorrow_value = np.array(rs_final_list[min(obs_index+1, len(rs_final_list) - 1)]['min_value'])
                            if previous_min < tomorrow_value:
                                yesterday_min_value = previous_min
                                tqdm.write(f"Skipping {nicedate(obs_date)} which is already at {previous_min}")
                                continue
                    else:
                        # Center bounds around last estimate
                        param_sources = np.stack([bounds_base[0], gsw_est_params, yesterday_params, previous_nss])
                        # bounds_radius = (np.max(param_sources, 0) - np.min(param_sources, 0)) / 2
                        # bounds = [np.array(parameter_start1) - bounds_radius, np.array(parameter_start1) + bounds_radius]
                        bounds = bounds_base
                        # parameter_start = rs_final_list[min(obs_index+1, len(rs_final_list))]['nss_params']
                        # parameter_start = rs_final_list[max(obs_index-1, 0)]['nss_params']
                        # parameter_start2 = clip_at_bounds(gsw_est_params)
                        parameter_start2 = clip_at_bounds(yesterday_params)
                        parameter_start1 = clip_at_bounds(previous_nss)
                        # parameter_start2 = clip_at_bounds(np.random.normal(parameter_start1, np.array([3, 15, 12, 12, 6, .1])))
                        # parameter_start3 = clip_at_bounds(np.random.normal(parameter_start1, np.array([3, 15, 12, 12, 6, .1])))
                else:
                    param_sources = np.stack([bounds_base[0], gsw_est_params, yesterday_params])
                    parameter_start1 = gsw_est_params
                    random_draw = np.random.uniform(bounds[0], bounds[1])
                    parameter_start2 = gsw_est_params + random_draw
                    bounds =[np.min(param_sources, 0), np.max(param_sources, 0)]
                    bounds_radius = (bounds[1] - bounds[0]) / 2


                if DO_ZERO_RIDGE:
                    rs_ql.regularize_prior = zeroPrior
                if FIT_METHOD == CRSPQuantlib.FitMethod.NELDER_MEAD:
                    # Start optim here
                    fits = []
                    # fits.append(rs_ql.fit_quickly(parameter_start1, bounds))
                    # fits.append(rs_ql.fit_quickly(parameter_start2, bounds))
                    # fits.append(rs_ql.fit_quickly(parameter_start3, bounds))


                    # Randomize around previous estimates
                    fits.append(rs_ql.fit_quickly(clip_at_bounds(add_rand(parameter_start1)), bounds_crazy))
                    fits.append(rs_ql.fit_quickly(clip_at_bounds(add_rand(parameter_start1)), bounds_crazy))
                    fits.append(rs_ql.fit_quickly(clip_at_outer_bounds(add_rand(previous_nss)), bounds_crazy))
                    fits.append(rs_ql.fit_quickly(clip_at_outer_bounds(add_rand(previous_nss)), bounds_crazy))

                    # Compare with ridge regression, as it typically has nice parameter estimates
                    fits.append(rs_ql.fit_quickly(clip_at_outer_bounds(add_rand(compare_estimates)), bounds_crazy))
                    # fits.append(rs_ql.fit_quickly(clip_at_outer_bounds(compare_estimates)))

                    # fits.append(rs_ql.fit_quickly(clip_at_outer_bounds(parameter_start2), bounds_crazy))
                    fits.append(rs_ql.fit_quickly(clip_at_outer_bounds(add_rand(parameter_start2)), bounds_crazy))


                    # fits.append(rs_ql.fit_quickly(clip_at_bounds(add_rand(bounds_mean)), bounds_crazy))
                    # for i in range(2):
                    #     # tqdm.write(f"Search {2*i+1} on date {nicedate(obs_date)} ")
                    #     fits.append(rs_ql.fit_quickly(clip_at_bounds(add_rand(parameter_start1)), bounds_crazy))
                    #     # tqdm.write(f"Search {2*i+2} on date {nicedate(obs_date)} ")
                    #     fits.append(rs_ql.fit_quickly(previous_nss), bounds_crazy)

                    best_fit = sorted(fits, key=lambda f: f['min_value'])[0]
                    rs_ql.assign_nss_params(best_fit['nss_params'])
                    fit = rs_ql.fit_quickly_finalize()
                    fit['min_value'] = best_fit['min_value']
                    fit['end_criterion'] = best_fit['end_criterion']
                else:
                    fit = rs_ql.fit(ql_array(parameter_start1), bounds)



                # tqdm.write(rs_ql.yieldcurve.fitResults().weights())

                # # If we hit bounds, try again
                # params = rs_ql.nss_params_out
                # if bounds is not None:
                #     hitBounds = [abs(p - low) < 1e-6 or abs(p  - high) < 1e-6 for (p, low, high) in zip(params, bounds[0], bounds[1])]
                # elif any(hitBounds):
                #     tqdm.write("Hit parameter bounds on " + nicedate(obs_date))
                #     if REFINE_ESTIMATES:
                #         # Recenter bounds to lie around the new start point
                #         bounds = [np.array(params) - bounds_radius, np.array(params) + bounds_radius]
                #         # We'll save this to disk to make sure we record the constraint issue
                #         previous_estimates['end_criterion'] = 50
                #         improved_estimates[obs_date] = previous_estimates
                #         fit = rs_ql.fit(params, bounds)
                #         fit['end_criterion'] = 50
                #         if any([abs(p - low) < 1e-6 or abs(p  - high) < 1.0e-6 for (p, low, high) in zip(rs_ql.nss_params_out, bounds[0], bounds[1])]):
                #             tqdm.write("Hit bounds twice!")
                #             fit['end_criterion'] = 100
                #     else:
                #         fit = rs_ql.fit(params, bounds)
                #         bounds = None # Unbounded

                # if (fit['min_value'] - yesterday_min_value) > .0001:
                #     if bounds is not None:
                #         bounds = [np.array(yesterday_params) - bounds_radius, np.array(yesterday_params) + bounds_radius]
                #     fit2 = rs_ql.fit(yesterday_params, bounds)
                #     if (fit['min_value'] - fit2['min_value']) > 10e-8:
                #         # tqdm.write("Improved fit by searching near yesterday's params. Moved from " + str(fit['min_value']) + " to " + str(fit2['min_value']))
                #         fit = fit2

                if REFINE_ESTIMATES:
                    pct_improvement = 1 - (fit['min_value'] / previous_min)
                    if pct_improvement > 0:
                        improved_estimates[obs_date] = fit
                        if pct_improvement > .005:
                            tqdm.write("Improved fit on {} by {:.2%} from {: f} to {: f}".format(
                                nicedate(obs_date), pct_improvement,
                                previous_min, fit['min_value']))
                    else:
                        fit['min_value'] = previous_min
                        # improved_estimates[obs_date] = previous_estimates
                        # tqdm.write("No improvement at date " + nicedate(obs_date))
                else:
                    rs.append(fit)
                    last_params = rs_ql.nss_params_out

                yesterday_min_value = fit['min_value']
            except Exception as e:
                tqdm.write("Calculation on date " + nicedate(obs_date) + " failed! Exception: " + str(e))
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
       rs = loadpklz(batch_file_path + str(batch + 1) + '.pklz')
       for est in rs:
           rs_final[est['obs_date']] = est
    savepklz(rs_final, destfile)


if DO_FINALIZE:
    CRSPResults = loadpklz(refinefile)
    CRSPdates = sorted(list(CRSPResults.keys()))
    CRSPResults_list = [CRSPResults[d] for d in CRSPdates]
    savepklz((CRSPdates, CRSPResults_list), refinefile_list)



# Some plotting functions
do_plots = False
if do_plots:
    lock = LockFile(refinefile)
    lock.acquire()
    rs_final = loadpklz(refinefile)
    print("Hello")
    rs_final_list = []
    rs_final_dates = []
    for d in selected_dates:
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


