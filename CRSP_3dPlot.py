
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

REFINE_ESTIMATES = True
DO_EQUAL_WEIGHT = False
DO_ZERO_RIDGE = False
NUM_ITER = 1

FIT_METHOD = CRSPQuantlib.FitMethod.NELDER_MEAD
# FIT_METHOD = CRSPQuantlib.FitMethod.ANNEALING
# FIT_METHOD = CRSPQuantlib.FitMethod.DE



if DO_ZERO_RIDGE:
    if DO_EQUAL_WEIGHT:
        refinefile = 'data/EWZeroRidge/CRSP_fit_results_refined.pklz'
    else:
        refinefile = 'data/ZeroRidge/CRSP_fit_results_refined.pklz'
else:
    if DO_EQUAL_WEIGHT:
        refinefile = 'data/EqualWeight/CRSP_fit_results_refined.pklz'
    else:
        refinefile = 'data/temp/CRSP_fit_results_refined.pklz'



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

zeroPrior = np.array([0, 0, 0, 0, 0, 0])

bounds_small = [np.array([      0,    -15,   -30,    -30, 1.0/2.5, 1/5.5]),
                np.array([     15,     30,    30,     30,     200, 1/2.5])]
bounds_base =  [np.array([      0,    -20,  -200,   -200,      .4,     0]),
                np.array([     15,    200,   200,    200,      60,    .4])]
bounds_crazy = [np.array([      0,  -9000, -9000,  -9000,       0,     0]),
                np.array([   9000,   9000,  9000,   9000,    5000,  5000])]

bounds_mean =   np.array([2.5, 2.5, 10, 10, 1, .2])

def clip_at_bounds(p):
    return np.clip(np.array(p), bounds_base[0], bounds_base[1])
def clip_at_outer_bounds(p):
    return np.clip(np.array(p), bounds_crazy[0], bounds_crazy[1])
def add_rand(p):
    return np.random.normal(p, np.array([1, 20, 20, 20, 6, .1]))


lock = LockFile(refinefile)
lock.acquire()
rs_final = loadpklz(refinefile)
lock.release()
rs_final_dates = sorted(list(rs_final.keys()))
rs_final_list = [rs_final[k] for k in rs_final_dates]
improved_estimates = {}



gsw_est_params = list(gsw_params.iloc[-1])



# Choose a date

# For 2d plot: two dates
# obs_date = np.datetime64(unique_dates[6100]) # In 2015
obs_date = np.datetime64(unique_dates[3112]) # in 2003
obs_date = np.datetime64(unique_dates[1095]) # in 1995

# Locate a date
# np.argmax(unique_dates_np == np.datetime64(pd.Timestamp('6/20/2003').to_datetime()))


# # For 3d plot
# obs_date = np.datetime64(unique_dates[5500])



print(nicedate(obs_date))



# Do estimation
keep_crsp_obs = cal_date == obs_date
npdata2 = npdata[keep_crsp_obs,:]
bond_tdatdt2 = bond_tdatdt[cal_date == obs_date]
obs_index = int(np.where(rs_final_dates == obs_date)[0])
previous_estimates = rs_final_list[obs_index]
previous_min = previous_estimates['min_value'] + .001
previous_nss = previous_estimates['nss_params']
yesterday_params = rs_final_list[max(obs_index-1, 0)]['nss_params']
tomorrow_params = rs_final_list[min(obs_index+1, len(unique_dates_np))]['nss_params']
gsw_est_params = list(gsw_params.loc[obs_date])


rs_ql = CRSPQuantlib.FitSecuritiesQuantlib(*[npdata2[:,c] for c in range(0,5)], bond_tdatdt2, obs_date)
rs_ql.equal_weight = DO_EQUAL_WEIGHT
rs_ql.fit_method = FIT_METHOD
if DO_ZERO_RIDGE: rs_ql.regularize_prior = zeroPrior


# Clear namespace for speed
del gsw_params, npdata, npdata2, rs_final, rs_final_dates, rs_final_list, unique_dates, unique_dates_np








### 2d price / yield curve fit comparison ####

rs_ql.assign_nss_params(previous_nss)


from scipy import interpolate
par_yield, dates = rs_ql.par_yields()
yield_plot_mats = np.array([rs_ql.opt.day_counter.yearFraction(rs_ql.obs_date_ql, d) for d in dates])


# For 1995 version
par_yield[0] = .058
par_yield[1] = .0585


par_spline_est = interpolate.UnivariateSpline(np.hstack([[-.05], yield_plot_mats, yield_plot_mats[-1] + 1]),
                                              np.hstack([max(par_yield[0] - .001, 0), par_yield, par_yield[-1]]), k=5, s=0.01)

yield_plot_mats_interp = np.arange(0.0, 10.01, 0.02)
par_yield_interp = par_spline_est(yield_plot_mats_interp)



import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams['axes.titlepad'] = 14
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
fig = plt.figure(0)




fig.clear()
axes = fig.add_subplot(1, 1, 1)
axes.plot(yield_plot_mats_interp, par_yield_interp * 100, 'k')
axes.plot(rs_ql.bond_plot_mats, rs_ql.bond_plot_yields * 100, 'b+')

# Potentially split bonds up
# axes.plot(rs_ql.bond_plot_mats[rs_ql.itype == 1], rs_ql.bond_plot_yields[rs_ql.itype == 1] * 100, 'r+')
# r = rs_ql.bond_coupon_rates
# axes.plot(rs_ql.bond_plot_mats[r > 9.0], rs_ql.bond_plot_yields[r > 9.0] * 100, 'r+')
# axes.plot(rs_ql.bond_plot_mats[np.logical_and(r > 0.0, r < 7.0)], rs_ql.bond_plot_yields[np.logical_and(r > 0.0, r< 7.0)] * 100, 'r+')


axes.set_xlabel("Maturity (years)", size="large")
axes.set_ylabel("Yield (%)", size="large")
axes.set_title(f"Par yields: {nicedate(obs_date)}", size="16")
axes.legend(["Par yield curve (estimated)", "Yield-to-maturity (observed)"], loc='lower right', fontsize="large")
fig.set_tight_layout(True)
plt.savefig(f"tabsandfigs/yieldcurve_{nicedate(obs_date).split('/')[-1]}.pdf", format='pdf', dpi=288)


fig.clear()
axes2 = fig.add_subplot(1, 1, 1)
axes2.scatter(rs_ql.bond_plot_mats, rs_ql.cleanPrices, s=5.0, c=[.5, .7, 1.0])
axes2.scatter(rs_ql.bond_plot_mats, rs_ql.bond_prices, s=0.5, c='k')
axes2.set_xlabel("Maturity (years)", size="large")
axes2.set_ylabel("Price (USD $1000)", size="large")
axes2.legend(["Observed", "Fitted"], loc='upper left', fontsize="medium")
axes2.set_title(f"Bond prices: {nicedate(obs_date)}", size="x-large")
fig.set_tight_layout(True)
plt.savefig(f"tabsandfigs/pricing_{nicedate(obs_date).split('/')[-1]}.pdf", format='pdf', dpi=288)


plt.close()




























################# 3d stuff starts here #################








# Center bounds around last estimate
param_sources = np.stack([bounds_base[0], gsw_est_params, yesterday_params, previous_nss])
bounds = bounds_base

parameter_start0 = np.random.uniform(bounds_small[0], bounds_small[1])
parameter_start1 = clip_at_bounds(gsw_est_params)
parameter_start2 = clip_at_bounds(yesterday_params)
parameter_start3 = clip_at_bounds(tomorrow_params)
parameter_start_previous = clip_at_bounds(previous_nss)



fits = []
fits.append(rs_ql.fit_quickly(parameter_start0, bounds_crazy))
fits.append(rs_ql.fit_quickly(parameter_start1, bounds_crazy))
fits.append(rs_ql.fit_quickly(parameter_start2, bounds_crazy))
fits.append(rs_ql.fit_quickly(parameter_start3, bounds_crazy))
fits.append(rs_ql.fit_quickly(clip_at_bounds(add_rand(parameter_start1)), bounds_crazy))
fits.append(rs_ql.fit_quickly(clip_at_outer_bounds(add_rand(previous_nss)), bounds_crazy))
fits.append(rs_ql.fit_quickly(clip_at_outer_bounds(parameter_start2), bounds_crazy))
fits.append(rs_ql.fit_quickly(clip_at_bounds(add_rand(parameter_start2)), bounds_crazy))



fit_rank = sorted(fits, key=lambda f: f['min_value'])

print([f['min_value'] for f in fit_rank])

for f in fit_rank:
    print(np.array(f['nss_params']))
    # print(f['nss_params'][0] + f['nss_params'][1])
    print(f['min_value'])


best_fit = fit_rank[0]

# fit = rs_ql.fit_quickly_finalize()
def get_fit(params):
    fit = rs_ql.assign_nss_params_quickly(list(params))
    return fit


sol_1 = np.array(fit_rank[0]['nss_params'])
sol_2 = np.array(fit_rank[4]['nss_params'])
sol_3 = np.array(fit_rank[5]['nss_params'])



# For 2013/01/04:
sol_1 = np.array([   4.62 ,   -4.622, -233.019,  226.466,    0.51 ,    0.517])
sol_2 = np.array([   4.803,   -4.727, -190.594,  195.1  ,    0.097,    0.093])
sol_3 = np.array([   0.,       0.076, -214.166,  228.306,    0.088,    0.083])


# Alt version
# sol_3 = [ 10.346, -10.27,   -0.07,  -12.157,   0.183,   0.145]
# sol_4 = [6.324, -6.249, -242.836, 244.297, 0.1, 0.097]


# Move from one solution to the next
dim1 = sol_2 - sol_1
dim2 = sol_3 - sol_1
dim2 = dim2 - dim1  # "Orthogonalize"
# dim2 = np.array([0, 0, 1, .1, 1, .1 ])




# 3d plot
gridpts_3d = 100
bounds1 = (-0.5, 1.1)
bounds2 = (-0.4, 1.3)


# # alt version
# gridpts_3d = 25
# bounds1 = (-0.5, 1.1)
# bounds2 = (-0.4, 1.25)


extent1 = np.linspace(*bounds1, gridpts_3d)
extent2 = np.linspace(*bounds2, gridpts_3d)
X, Y = np.meshgrid(extent1, extent2)


res_matrix = np.zeros((gridpts_3d, gridpts_3d))
with tqdm(total=gridpts_3d * gridpts_3d) as progress:
    for i1, e1 in enumerate(extent1):
        for i2, e2 in enumerate(extent2):
            # res_matrix[i1, i2] = get_fit(clip_at_outer_bounds(sol_1 + dim1 * e1 + dim2 * e2))
            res_matrix[i1, i2] = get_fit((sol_1 + dim1 * e1 + dim2 * e2))
            progress.update(1)

print(np.max(res_matrix))

# fit = rs_ql.fit_quickly_finalize()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

mpl.rcParams['grid.color'] = '0.9'
mpl.rcParams['grid.linewidth'] = '1.0'
mpl.rcParams['axes.titlepad'] = '24'

fig = plt.figure(0)
fig.clf()
ax3d:Axes3D = fig.add_subplot(111, projection='3d', proj_type='ortho')



ax3d.cla()
ax3d.view_init(30, 145)
surf = ax3d.plot_surface(X, Y, res_matrix, color="0.3", zorder=5, alpha=0.1, rcount=30, ccount=30)
ax3d.plot_wireframe(X, Y, res_matrix, color="0.50", lw=0.5, rcount=25, ccount=25, zorder=5)
ax3d.plot([0, .001], [0, .001], [0, 7e5], c="k", lw=1.5, linestyle="-", zorder=1)
ax3d.text(0, 0, 8e5, "M0", color="k", fontsize=10, horizontalalignment='center', zdir=None)

ax3d.plot( [0, 0.001], [1.0, 1.01], [0, 6e5], c="k", lw=1.5, linestyle="-", alpha=1, zorder=3)
ax3d.text(0, 1, 7e5, "M1", color="k", fontsize=10, horizontalalignment='center', zdir=None)


ax3d.plot([1, 1.001], [1, 1.001], [6.5e5, 7.5e5], c="k", lw=1.5, linestyle=":", zorder=0)
ax3d.text(1, 1, 8e5, "M2", color="k", fontsize=10, horizontalalignment='center', zdir=None)
ax3d.plot([1, 1.001], [1, 1.001], [0, 6.4e5], c="k", lw=1.5, linestyle=":", alpha=0.4, zorder=0)
plt.draw()


get_fit(sol_1)
get_fit(sol_2)
get_fit(sol_3)



ax3d.tick_params(colors ='0.1')

ax3d.set_xticks(np.arange(-0.4, 1.1, .2), False)
ax3d.set_xticklabels([''] * 2 + ['0.0'] + [''] * 4 + ['1.0'])

ax3d.set_yticks(np.arange(-0.4, 1.3, .2), False)
ax3d.set_yticklabels([''] * 2 + ['0.0'] + [''] * 4 + ['1.0'])
ax3d.tick_params(direction='out', pad=-1, color='0.2')
ax3d.set_zlim([-5.0, ax3d.get_zlim()[1]])


ax3d.tick_params(axis='z', pad=2.5, color='0.4')
ax3d.set_zticklabels(['0.0'] + [''] * 4 + [r'$2\cdot10^{5}$'])

ax3d.xaxis.line.set_color('0.4')
ax3d.yaxis.line.set_color('0.4')
ax3d.zaxis.line.set_color('0.4')


ax3d.set_title("NSS objective function 1/4/2012", fontsize=12)
# fig.suptitle("")
# fig.suptitle("Nelder-Mead stopping points", y=.975, fontsize=14)
# fig.tight_layout(rect=[0, 0.03, 1, 0.925])
fig.tight_layout(rect=[-.02, 0.0, 1.02, 0.98])

plt.savefig("tabsandfigs/Objective3d_wide.pdf", format='pdf')







# Alt version
# ax3d.cla()
# res_matrix2 = res_matrix
# res_matrix2[res_matrix > 2.0e6] = np.nan
# ax3d.plot_surface(X, Y, np.minimum(res_matrix, 4.0e6))
# ax3d.plot_surface(X, Y, res_matrix)
# ax3d.view_init(35, 355)
# plt.draw()
# ax3d.plot_wireframe(X, Y, np.minimum(res_matrix, 4.0e6), rcount = 40, ccount = 40)
# ax3d.set_zlim3d(0, 4e6)

# ax3d.plot([0, .001], [0, .001], [0, 7e5], c="k", lw=1.5, linestyle="-", zorder=1)
# ax3d.plot([0, 0.001], [1.0, 1.01],  [0, 2e6], c="k", lw=1.5, linestyle="-", alpha=1, zorder=1)
# ax3d.plot([1, 1.001], [1, 1.001], [0, 1e6], c="k", lw=1.5, linestyle="-", zorder=0)

# ax3d.cla()
# scatter_pt = ax3d.scatter(0, 0, get_fit(sol_1), c="k", zorder=10, depthshade=False)
# scatter_pt = ax3d.scatter(0, 0, get_fit(sol_1), c="r", zorder=20, depthshade=False)
# ax3d.plot_surface(X, Y, res_matrix, zorder=5)
# scatter_pt = ax3d.scatter(0, 0, get_fit(sol_1), c="r", zorder=30, depthshade=False)
# scatter_pt = ax3d.scatter(0, 0, get_fit(sol_1), c="r", zorder=0, depthshade=False)
# scatter_pt = ax3d.plot([0, 0], [0, 0], [get_fit(sol_1), get_fit(sol_1) + 10], c="r", lw=5, linestyle="--", zorder=10)
# scatter_pt = ax3d.plot3D([0], [0], [1], c="r", zorder=1)




# Look at yield curves. Not so different!
# quickplot(nss_yield(sol_1))
# qpadd(nss_yield(sol_2))
# quickplot(nss_yield(clip_at_outer_bounds(sol_1 + dim1*-.01)))


################# 3d stuff starts here #################

