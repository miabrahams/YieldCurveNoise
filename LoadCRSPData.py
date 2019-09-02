
from mga_helpers import *

import pandas as pd
import numpy as np
import zipfile
import sqlite3
from tqdm import tqdm

# For scratch work
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import datetime
pd.options.display.float_format = '{:,.2f}'.format # Reduce decimal points to 2



## Update files: Convert from CSV to pickle
def crsp_csv_to_pickle():
    zf = zipfile.ZipFile('data/CRSP_Treasury_Issues_v2.zip')
    CRSPData = pd.read_csv(zf.open('CRSP_Treasuries.csv'))
    zf.close()
    make_date_column(CRSPData, 'TMATDT')
    make_date_column(CRSPData, 'TDATDT')
    make_date_column(CRSPData, 'TFCPDT')
    make_date_column(CRSPData, 'CALDT')
    CRSPData['BASPREAD'] = CRSPData.TDASK - CRSPData.TDBID
    CRSPData['TDYLD']    = CRSPData['TDYLD'] * 100 * 365
    CRSPData['DIRTYPRC'] = CRSPData.TDNOMPRC + CRSPData.TDACCINT
    CRSPData['ttm']      = (CRSPData.TMATDT - CRSPData.CALDT).dt.days  # Convert to days
    CRSPData = CRSPData[CRSPData.TDNOMPRC != 0]
    CRSPData.to_pickle("data/CRSP_Treasuries.pkl.compress", compression="gzip")
    return CRSPData

CRSPData = crsp_csv_to_pickle()
CRSPData = pd.read_pickle("data/CRSP_Treasuries.pkl.compress", compression="gzip")




# Save info per-cusip to use in GovPX estimation
def crsp_by_cusip():
    CRSP_by_cusip = CRSPData.groupby('TCUSIP').last()
    savepklz(CRSP_by_cusip, 'data/CRSP_by_cusip.pklz')


## Load pickled data and create SQLite database.
## Note for intraday use: we may want to store the dates as INTEGER type in the SQL database, to allow
## range-based comparisons.
## NOTE: zip this file up for later use
def crspdata_to_sqlite(CRSPData):
    conn = sqlite3.connect('data/CRSPSQL')
    CRSPData.to_sql('CRSP', conn, if_exists='replace')
    conn.close()
crspdata_to_sqlite(CRSPData)


# Create a list of all observation dates. Takes a long time so we do it with a loop to get a progress bar.
# Interestingly this can't be improved by testing for equality first
# and only doing an append when the date increments. Why is that?
def get_unique_dates(CRSPData):
    unique_dates = set()
    for i in tqdm(range( CRSPData.CALDT.size)):
        unique_dates.add(CRSPData.CALDT[i])
    unique_dates = list(unique_dates)
    unique_dates.sort()
    unique_dates = pd.Series(unique_dates)
    unique_dates.to_pickle("data/CRSPUniqueDates.pkl.compress", compression="gzip")
    return unique_dates
unique_dates = get_unique_dates(CRSPData)
unique_dates = pd.read_pickle("data/CRSPUniqueDates.pkl.compress", compression="gzip")


## CRSP data description ##
# crsp_columns = ["KYTREASNO", "CRSPID", "TCUSIP", "TMATDT", "IWHY", "TCOUPRT", "TVALFC", \
#                 "TFCPDT", "ITYPE", "CALDT", "TDBID", "TDASK", "TDNOMPRC", "TDACCINT", "TDDURATN"]
# KYTREASNO is Treasury record identifier
# CRSPID is CRSP-assigned unique identifier
# KYCRSPID is same thing but in a SAS format
# TCUSIP is CUSIP
# TMATDT is maturity date
# IWHY is reason code for series terminating
# TCOUPRT is coupon rate
# TVALFC is amount of first coupon
# TFCPDT is first coupon payment date. TMATDT is maturity date.
# IFCPDTF is fcp date flag: -1 for estimated, 1 for verified, 0 for no coupon
# CALDT is date of observation. Set this as date column.
# TDBID/TDASK are EOD bid/ask quotes
# TDACCINT is accrued interest
# TDYLD is computed yield to maturity
# TDTOTOUT is total outstanding in millions $USD
# TDPDINT is paid interest. Usually zero.
# TDDURATN is Macaulay duration
# CRSP database is organized by KYTREASNO/TCUSIP.

# ITYPE
# 1 = Noncallable bond
# 2 = Noncallable note
# 3 = Certifcate of indebtedness
# 4 = Treasury Bill
# 5 = Callable bond
# 6 = Callable note
# 7 = Tax Anticipation Certifcate of Indebtedness
# 8 = Tax Anticipation Bill
# 9 = Other, ﬂags issues with unusual provisions
# 10= Reserved for future use
# 11= Inﬂation-Adjusted Bonds
# 12= Inﬂation-Adjusted Notes



CRSPData = pd.read_pickle("data/CRSP_Treasuries.pkl.compress", compression="gzip")

# Load a slice
query_date = '1994-02-01'

# Use SQLite database (broken atm, need to use .zip file)
def get_df_sqlite(query_date):
    conn = sqlite3.connect('data/CRSPSQL')
    query_vars = 'CALDT, KYTREASNO, TCUSIP, TMATDT, TCOUPRT, TVALFC, TFCPDT, ITYPE, TDBID, TDASK, TDNOMPRC, TDDURATN, TDACCINT, TDYLD, DIRTYPRC, BASPREAD, maturity, ttm, caldt'
    df = pd.read_sql_query('select ' + query_vars + ' from CRSP where CALDT == "' + query_date + '"', conn)
    make_date_column(df, 'CALDT', '%Y-%m-%d %H:%M:%S')
    not_inflation_indexed = (df.ITYPE != 12) & (df.ITYPE != 11)
    not_callable = (df.ITYPE != 5) & (df.ITYPE != 6)
    not_excluded = ~(df.TCUSIP.isin(bad_CUSIPs))
    df_export = df[not_inflation_indexed & not_callable & not_excluded].copy(deep=True)
    df.export = df_export.reset_index(drop=True)
    return df_export
df_export = get_df_sqlite(query_date)


# Just match in memory
from CRSP_Helpers import *
df_export = get_df_inplace(CRSPData, query_date)

df_export.iloc[67]



# Write pickle
df_export.to_pickle("data/daily_cross_section.pkl.compress", compression="gzip")


# Write database
def write_df_sqlite(df_export):
    df_export = df_export[['CALDT', 'TDNOMPRC', 'TMATDT', 'TCOUPRT', 'TCUSIP', 'TVALFC', 'TFCPDT', 'ITYPE', 'TDACCINT', "TDYLD", "BASPREAD", "TDATDT"]].copy()
    df_export['TFCPDT'] = df_export['TFCPDT'].astype(str)
    export_conn = sqlite3.connect('data/daily_cross_section')
    df_export.to_sql('daily_cross_section', export_conn, if_exists='replace')
    export_conn.close()
write_df_sqlite(df_export)


# Load GSW yields


# Load Gurkaynak, Sack, Wright dataset.
# These data are extracted from here: https://www.federalreserve.gov/pubs/feds/2006/200628/200628abs.html
data = pd.read_excel('data/gsw_ns_params.xlsx', parse_dates=[0])
gsw_params = pd.DataFrame({"beta0": data['BETA0'] / 100.0,
                           "beta1": data['BETA1'] / 100.0,
                           "beta2": data['BETA2'] / 100.0,
                           "beta3": data['BETA3'] / 100.0,
                           "kappa0": np.reciprocal(data['TAU1']),
                           "kappa1": np.reciprocal(data['TAU2']),
                           "date": data.Date})
gsw_params.set_index("date", inplace=True)
savepklz(gsw_params, "data/daily_gsw_params.pklz")

gsw_params: pd.DataFrame = loadpklz("data/daily_gsw_params.pklz")





##### SCRATCH

# Plot time to maturity
df = df_export
plt.clf()
date_mat = pd.DatetimeIndex(df.TMATDT).to_pydatetime() # pydatetime is best for matplotlib
date_obs = pd.DatetimeIndex(df.CALDT).to_pydatetime()
fig, axes = plt.subplots()
plt.scatter(date_mat, np.ones(df.TMATDT.shape))
axes.xaxis.set_major_locator(mdates.AutoDateLocator())
axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate()
plt.axvline(pd.DatetimeIndex(df.CALDT).to_pydatetime()[0] + datetime.timedelta(20*365), color='r')
plt.scatter(df['TDDURATN'] / 365.0, df['TDDURATN'] / 365.0)


plt.plot_date(df['TMATDT'][yld_avail], df['TDYLD'][yld_avail], xdate=True)
max(CRSPData.CALDT)
CRSPData
d = CRSPData.set_index(['CALDT', 'TCUSIP'])
d.sort_index(level=True)
LastDay = d.loc[np.datetime64('2017-12-29 00:00:00')]
np.datetime64('2017-12-29 00:00:00')

odd_bond = CRSPData[CRSPData.TCUSIP == '']
odd_bond.plot(x="CALDT", y="TDYLD")

next_bond = CRSPData[CRSPData.TCUSIP == '912827V7']
next_bond.plot(x="CALDT", y="TDYLD")





#  OLDER CRAP THAT MIGHT BE USEFUL AFTER THIS LINE #


help(pd.to_pickle)

# crsp_monthly.date = pd.to_datetime(crsp_monthly.date)
# compustat_annual.fyear = pd.to_numeric(compustat_annual.fyear, downcast='int') # Got some NaNs here?

# Construct factors from CRSP
# crsp_monthly['mve'] = np.log(crsp_monthly.PRC * crsp_monthly.SHROUT) #log market value equity
# crsp_monthly['dy'] = crsp_monthly.DIVAMT.div(crsp_monthly.SHROUT * crsp_monthly.PRC) # Dividend yield
# crsp_monthly['']


# Load data
# RPSData = pd.read_feather("data/RPSData_RFS.feather", nthreads=8)
# RPSVariables = RPSData.dtypes
# RPSData.iloc(0)
#
# # Create calendar year from Compustat fiscal year
# RPSData['DATE'] = pd.to_datetime(RPSData['DATE'], format='%Y%m%d', errors='ignore')
#
# # Do some OLS
# RPSData[['chmom', 'disp']]
# model = smf.ols('RET ~ chmom + pchquick', data=RPSData)
# res = model.fit()
# print(res.summary())
#
# # Construct pivot table
# crsp_monthly_pivot = crsp_monthly.pivot_table(values=['PRC', 'VOL', 'SHROUT',  'DIVAMT',  'CUSIP'], index=['date'], columns=['PERMNO'])
#
# #Merge annual price / volume
# pd.DatetimeIndex(RPSData.datadate).quarter
#
# RPSData.dtypes
# RPSData.columns
#
# crsp_monthly.index =
# crsp_monthly_eoy = crsp_monthly[pd.DatetimeIndex(crsp_monthly.date).month == 12]
#
# Compustat_annual_crsp =
#
# crsp_monthly.dtypes
#
# RPSData.dtypes
# RPSData.ib
# RPSData.head(1)