import pandas as pd
import QuantLib as ql
from tqdm import tqdm
import importlib
import CRSPQuantlib
from CRSP_Helpers import *
from mga_helpers import *
import numpy as np

# Parallel evaluation imports
from dask import delayed, compute
import dask.multiprocessing
from dask.diagnostics import ProgressBar
# Be sure to install 'distributed' package
from dask.distributed import Client
# from distributed import LocalCluster
from dask.distributed import LocalCluster
import time


importlib.reload(CRSPQuantlib)

CRSPData = pd.read_pickle("data/CRSP_Treasuries.pkl.compress", compression="gzip")
first_cusip_issue = pd.read_pickle("data/first_issue_date.pkl.compress", compression="gzip")
unique_dates = pd.read_pickle("data/CRSPUniqueDates.pkl.compress", compression="gzip")
CRSPData = clean_bad_bonds(CRSPData)
first_cusip_issue_dict = dict(zip(first_cusip_issue.index, first_cusip_issue.IssueDate))


# Gurkaynak, Sack and Wright starting values for parameters
gsw_params: pd.DataFrame = loadpklz("data/daily_gsw_params.pklz")



#### Prep for parallelization.
# Pandas hogs the GIL so split data into numpy matrices.
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



# Helper function to compute estimates for a single batch of dates
def get_fit_results_grp(npdata, first_cusip_issue_dict, cal_date, tdatdt, dates):
    res_out = []
    for j in range(len(dates)):
        date = dates[j]
        print("Calculation on date " + str(date)[:10])
        current_obs = (cal_date == date)
        npdata2 = npdata[current_obs, :]
        # Note: takes about 1-1.5 sec running on a single core.
        try:
            res_out.append(CRSPQuantlib.fit_securities_quantlib(npdata2[:, 0],  # bond_maturities
                                                                npdata2[:, 1],  # itype
                                                                npdata2[:, 2],  # bond_rates
                                                                npdata2[:, 3],  # bond_prices
                                                                npdata2[:, 4],  # cusips
                                                                tdatdt[current_obs],  # bond_tdatdt
                                                                date))
        except Exception as e:
            res_out.append("Calculation on date " + str(date)[:10] + " failed! Exception: " + str(e))
    return res_out



# Testing: run on one or more observations.
# If something goes wrong in an interactive session reload your data!
t = time.clock()
test_obs = [6000]
rs_test = get_fit_results_grp(npdata, first_cusip_issue_dict, cal_date, bond_tdatdt, unique_dates_np[test_obs])
print(time.clock() - t)
dates, fit_err = zip(*[(r['obs_date'], r['fit_err']) for r in rs_test])
fit_err_plot = pd.Series({'fit_err': fit_err}, index=dates)
fit_err_plot.plot(style='k')


# More Testing
importlib.reload(CRSPQuantlib)

# obs_date = unique_dates_np[6000]
# obs_date = pd.to_datetime('06/02/1994').to_datetime64()
obs_date = pd.to_datetime('December 02, 1994').to_datetime64()
npdata2 = npdata[cal_date == obs_date,:]
start_params = list(gsw_params.loc[obs_date])
rs_ql = CRSPQuantlib.FitSecuritiesQuantlib(npdata2[:,0], npdata2[:,1], npdata2[:,2], npdata2[:,3], npdata2[:,4], bond_tdatdt[cal_date == obs_date], obs_date)
rs_ql.fit()
rs_ql.plot_par_curve_fit()


res = rs


rs_ql.plot_fit_yields()
rs_ql.plot_fit_prices()

# Helper function to aggregate
def collect_fit_results(A_Fut, client):
    res = []
    failedWorkers = []
    for a, z in tqdm(zip(A, range(len(A_Fut)))):
        try:
            res.extend(client.gather(a))
        except:
            res.extend([])
            failedWorkers.append((z, len(res)))
            print("Worker failed: " + str(z))
    print(res)
    return res, failedWorkers



# Try using the distributed calculation method
c = LocalCluster(n_workers=10, processes=True)
# Scale up cores slowly - jumping straight to 40 made Dask unhappy
time.sleep(3)
c.scale_up(20)
time.sleep(3)
c.scale_up(30)
client = Client(c)


# Using client.scatter was a bit flaky, don't bother with it
remote_cusip_dict = []
# remote_cusip_dict = client.scatter(first_cusip_issue_dict)
# remote_npdata = client.scatter(npdata)
# remote_tdatdt = client.scatter(bond_tdatdt)
# remote_cal_date = client.scatter(cal_date)

# Split into more cores than necessary
n_obs = 6749
n_cores = 60
obs_per_worker = int(n_obs / n_cores)
worker_splits = [slice(j, n_obs, n_cores) for j in range(n_cores)]
worker_splits_test = [slice(j, n_obs, n_cores*10) for j in range(n_cores)]

A = []
t = time.clock()
for j in worker_splits_test:
    match_date = cal_date.isin(unique_dates_np[j])
    remote_npdata = npdata[match_date, :].copy()
    remote_tdatdt = bond_tdatdt[match_date].copy()
    remote_cal_date = cal_date[match_date].copy()
    A.append(client.submit(get_fit_results_grp, remote_npdata, remote_cusip_dict,
                           remote_cal_date, remote_tdatdt, unique_dates_np[j]))



res, failedWorkers = collect_fit_results(A, client)
print(time.clock() - t)


# We have a few observations that don't want to play nicely.
# bad_dates = np.where([type(r[6]) != np.float64 for r in res])
# bad_dates = [2061, 2740, 4477, 4484, 4487]


# Store the results as zipped pickle

savepklz(res, 'data/CRSP_fit_result.pklz')




# Clean up any failures with a second round (currently unused)
# client.close()
# c.close()
# c = LocalCluster(n_workers=5, processes=True, threads_per_worker=5)
# client = Client(c)
# B = []
# t = time.clock()
# for j in [worker_splits[f[0]] for f in failedWorkers]:
#     match_date = cal_date.isin(unique_dates_np[j])
#     remote_npdata = npdata[match_date, :].copy()
#     remote_tdatdt = bond_tdatdt[match_date].copy()
#     remote_cal_date = cal_date[match_date].copy()
#     B.append(get_fit_results_grp(remote_npdata, remote_cusip_dict, remote_cal_date, remote_tdatdt, unique_dates_np[j]))
#     print(j)
# print(time.clock() - t)


# res_final = []
# failedWorkers = []
# t = time.clock()
# for a, z in zip(A, range(len(A))):
#     try:
#         res_final.extend(client.gather(a))
#     except:
#         res_final.extend([])
#         failedWorkers.append((z, len(res_final)))
#         print("Worker failed: " + str(z))
# print(res_final)
# res = client.gather(B)
# print(time.clock() - t)
# savepklz(res_final, 'data/result_final.pklz')


# Serial computation option
# t = time.clock()
# res = []
# for j in tqdm(range(len(unique_dates_np))):
#     res.append(get_fit_results_grp(npdata, first_cusip_issue_dict, cat_date, unique_dates_np[j]))
# print(time.clock() - t)













# Load previous results
res = loadpklz('data/CRSP_fit_result.pklz')


# Aggregate results
dates = [r['obs_date'] for r in res]
fit_err = [r['fit_err'] for r in res]
fit_err_plot = pd.DataFrame({'fit_err': fit_err}, pd.DatetimeIndex(dates))
# TODO: Clean this plot up and prep it for the paper
fit_err_plot.resample("W").plot(y='fit_err', style='k-')
fit_err_plot.plot()

# TODO: Make charts of means and variances etc

caldt       = [r['obs_date'] for r in res]
cusips      = [r['cusips'] for r in res]
bond_prices = [r['bond_prices'] for r in res]
yieldcurves = [r['yieldcurve'] for r in res]

def plot_fitted_yield(yieldcurve):
    quickplot(yieldcurve[:, 2], yieldcurve[:, 0])

plot_fitted_yield(yieldcurves[0])
rs_test
