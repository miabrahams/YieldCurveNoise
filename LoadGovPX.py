
import pandas as pd
import numpy as np
import os
import zipfile
from mga_helpers import *
from CRSP_Helpers import *
import matplotlib.pyplot as plt

CRSP_by_cusip = loadpklz('data/CRSP_by_cusip.pklz')
# GovPX PKLZ file is available later

# CRSP_by_cusip.loc['912827L8']


# Missing data 10/29/2012 due to Hurricane Sandy
# Visible on every observation
# cusip # active # alias # ltside # date # matdate # time # seqno # rtc # coupon
# Not always available.
# bidprc # bidyld # bidsize # askprc # askyld # asksize # ltprc # ltyld # ltsize # vol


types = {  'ACTIVE': 'category', 'ALIAS': 'category', 'ASKPRC': 'float64', 'ASKSIZE': 'float64', 'ASKYLD': 'float64',
           'BIDPRC': 'float64', 'BIDSIZE': 'float64', 'BIDYLD': 'float64', 'CHWPRC': 'float64', 'CHWSIZE': 'float64',
           'CHWYLD': 'float64', 'COUPON': 'float64', 'CTWPRC': 'float64', 'CTWSIZE': 'float64', 'CTWYLD': 'float64',
           'CUSIP': 'category', 'INDASK': 'float64', 'INDAYLD': 'float64', 'INDBID': 'float64', 'INDBYLD': 'float64',
           'LTPRC': 'float64', 'LTSIDE': 'object', 'LTSIZE': 'float64', 'LTYLD': 'float64', 'RTC': 'float64',
           'SEQNO': 'float64', 'TIME': 'object', 'VOL': 'float64', 'MATDATE': 'category', 'DATE': 'category'}

def load_govpx(filename):
    """Read zipped csv of GovPX data. Converts column names to lowercase."""
    def _parse_date(date, time):
        return pd.to_datetime(date + " " + time, format="%d%b%Y %X.%f")
    file = zipfile.ZipFile(filename)
    innerfilename = filename.split('/')[-1].split('.')[0] + '.csv'
    # dat = pd.read_csv(file.open(innerfilename), dtype=types, parse_dates={'obs_time': ['DATE', 'TIME']}, date_parser = _parse_date, infer_datetime_format=True)
    dat = pd.read_csv(file.open(innerfilename), dtype=types)
    if dat.columns.contains('SEQNO'):
        dat.rename(mapper = lambda s: s.lower(), axis=1, inplace=True)
    dat.set_index('seqno')
    file.close()
    return dat


# Find GovPX files
base_path = 'data/GovPX/Original_Zip_Files/'
files = [f for f in os.listdir(base_path) if f.endswith(".zip")]
# sel = slice(4, len(files), 5) # Split across a few python instances
sel = slice(len(files)) # Do all at once


for f in files[sel]:
    print("Processing file " + f)
    filename = base_path + f
    outfile = base_path + f.split('.')[0] + '.pklz'
    if os.access(outfile, os.F_OK):
        continue   # This file has been processed already
    GovPX = load_govpx(filename)
    # Cleaning starts here
    GovPX.ix[GovPX.askprc == 0, 'askprc'] = np.nan
    GovPX.ix[GovPX.bidprc == 0, 'bidprc'] = np.nan
    has_price = (~(pd.isna(GovPX.bidprc)) | ~(pd.isna(GovPX.askprc))) # Keep only obs with bid & ask
    GovPX = GovPX[has_price].copy(deep=True)
    make_date_column_infer(GovPX, 'date') # Convert date columns
    make_date_column_infer(GovPX, 'matdate')
    GovPX.time = pd.to_datetime(GovPX.date.dt.strftime("%Y-%m-%d") + " " + GovPX.time, format="%Y-%m-%d %X")
    GovPX.cusip = GovPX.cusip.str[0:8] # Remove checksum digit
    GovPX = GovPX[~(GovPX.cusip.isin(bad_CUSIPs))] # This list is in CRSP_helpers.py
    tradeHour = GovPX.time.dt.hour
    GovPX = GovPX[(tradeHour >= 9) & (tradeHour <= 17)] # Fleming does 7:30 to 5. Doesn't matter for end-of-day obs.
    savepklz(GovPX, outfile)
    print("Processed file " + str(f))

pd.Timedelta()

for f in files[sel]:
    print("processing file " + f)
    filename = base_path + f
    outfile = base_path + 'Indicative/' + f.split('.')[0] + '.pklz'
    if os.access(outfile, os.F_OK):
        continue  # This file has been processed already
    GovPX = load_govpx(filename)
    GovPX.dropna(subset=['time'], inplace=True)
    pd.Timedelta(GovPX.time[0])
    make_date_column_infer(GovPX, 'date') # Convert date columns
    make_date_column_infer(GovPX, 'matdate')
    GovPX.time = pd.to_datetime(GovPX.date.dt.strftime("%Y-%m-%d") + " " + GovPX.time, format="%Y-%m-%d %X")





# ### LOAD PKLZ ###
GovPX = loadpklz("data/GovPX/ticks94.pklz")



# Look at time series per cusip
cusips = GovPX.cusip.unique()
cusip_test = GovPX.ix[GovPX.cusip == cusips[3]]
cusip_test = GovPX.ix[GovPX.cusip == '912810DV']
cusip_test = GovPX.ix[GovPX.cusip == '912795PJ']  # ?????
cusip_test = GovPX.ix[GovPX.cusip == '912795QB']
cusip_test = GovPX.ix[GovPX.cusip == '9128274Y']
cusip_test = GovPX.ix[GovPX.cusip == '912810CX']
cusip_test.plot(x='time', y=['bidprc', 'askprc'])


# Split by 10-min intervals
def trade_resampler(sample):
    return 0





# For testing
sample = GovPX.head(300000)

# Find last observation with bid price by cusip
# sample.groupby(sample.cusip)
cusips = sample.cusip.unique()
by_cusip = sample[sample.cusip == cusips[4]][['time', 'bidprc', 'askprc']]
by_cusip.askprc[by_cusip.askprc == 0] = np.nan
by_cusip.bidprc[by_cusip.bidprc == 0] = np.nan
by_cusip.plot(x='time')
by_cusip.shape


# Plot with weekends
x = by_cusip.time
y = by_cusip.bidprc
y2 = by_cusip.askprc
f,(ax,ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')

# plot the data on both axes. Split up by day to disconnect plots
for d in x.dt.date.unique():
    subset = x.dt.date == d
    ax.plot(x[subset], y[subset], 'r')
    ax.plot(x[subset], y2[subset], 'b')
    ax2.plot(x[subset], y[subset], 'r')
    ax2.plot(x[subset], y2[subset], 'b')


# Set the axes for week1 and week2
week1 = x[x.dt.week == 1]
week2 = x[x.dt.week == 2]
ax.set_xlim(week1.iloc[0], week1.iloc[-1])
ax2.set_xlim(week2.iloc[0], week2.iloc[-1])

# hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.yaxis.tick_left()
ax.tick_params(labelright='off')
ax2.yaxis.tick_right()
# Add pretty diagonal slashes
d = .015 # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False) # arguments to pass plot
ax.plot((1-d,1+d), (-d,+d), **kwargs)
ax.plot((1-d,1+d),(1-d,1+d), **kwargs)
kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d,+d), (1-d,1+d), **kwargs)
ax2.plot((-d,+d), (-d,+d), **kwargs)

import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%m/%d')
DailyDateLocator = mdates.DayLocator()
ax.xaxis.set_major_formatter(myFmt)
ax2.xaxis.set_major_formatter(myFmt)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax2.xaxis.set_major_locator(mdates.DayLocator())

plt.setp(ax.get_xticklabels(), rotation=45)
plt.setp(ax2.get_xticklabels(), rotation=45)
plt.subplots_adjust(bottom = 0.15)
plt.legend(['bidprc', 'askprc'])

ax.set_title("Week of 1-02-1994")
ax2.set_title("Week of 1-09-1994")


# Try this to compress space between days:
# https://www.reddit.com/r/learnpython/comments/4lh841/matplotlib_how_to_avoid_displaying_xaxis_entries/









#### ATTEMPTING TO LOOK AT ICAP DATA HERE #######





# Load other junk
def load_govpx(filename):
    """Read zipped csv of GovPX data. Converts column names to lowercase."""
    file = zipfile.ZipFile(filename)
    innerfilename = filename.split('/')[-1].split('.')[0] + '.csv'
    dat = pd.read_csv(file.open(innerfilename))
    if dat.columns.contains('SEQNO'):
        dat.rename(mapper = lambda s: s.lower(), axis=1, inplace=True)
    dat.set_index('seqno')
    file.close()
    return dat


filename = "C:/Users/miabr/Desktop/ticks09.zip"
innerfilename = filename.split('/')[-1].split('.')[0] + '.csv'
file = zipfile.ZipFile(filename)
chunks = pd.read_csv(file.open(innerfilename), chunksize=10**8)
dat = chunks.__next__()

for dtype in ['float','int','object', 'category', 'datetime64']:
    selected_dtype = dat.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))


# This gives us a list of columns
dtypes = dat.dtypes
dtypes_type = [i.name for i in dtypes.values]
column_types = dict(zip(dtypes.index, dtypes_type))
import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(column_types)

types = {'askprc': 'float64',
         'asksize': 'float64',
         'askyld': 'float64',
         'bidSize': 'float64',
         'bidprc': 'float64',
         'bidyld': 'float64',
         'coupon': 'float64',
         'cusip': 'category',
         'indask': 'float64',
         'indayld': 'float64',
         'indbid': 'float64',
         'indbyld': 'float64',
         'ltprc': 'float64',
         'ltsize': 'float64',
         'ltyld': 'float64',
         'matdate': 'category',
         'type': 'float64'}


def parse_date(date, time):
    return pd.to_datetime(date + " " + time, format="%d%b%Y %X.%f")

chunks = pd.read_csv(file.open(innerfilename),
                     dtype=types,
                     parse_dates={'obs_time': ['date', 'time']},
                     date_parser = parse_date,
                     infer_datetime_format=True,
                     chunksize=10**8)


dat = chunks.__next__()


dat.date + " " + dat.time

GovPX = load_govpx(filename)
# Cleaning starts here
GovPX.ix[GovPX.askprc == 0, 'askprc'] = np.nan
GovPX.ix[GovPX.bidprc == 0, 'bidprc'] = np.nan
has_price = (~(pd.isna(GovPX.bidprc)) | ~(pd.isna(GovPX.askprc))) # Keep only obs with bid & ask
GovPX = GovPX[has_price].copy(deep=True)
make_date_column_infer(GovPX, 'date') # Convert date columns
make_date_column_infer(GovPX, 'matdate')
GovPX.time = pd.to_datetime(GovPX.date.dt.strftime("%Y-%m-%d") + " " + GovPX.time, format="%Y-%m-%d %X")
GovPX.cusip = GovPX.cusip.str[0:8] # Remove checksum digit
GovPX = GovPX[~(GovPX.cusip.isin(bad_CUSIPs))] # This list is in CRSP_helpers.py
tradeHour = GovPX.time.dt.hour
GovPX = GovPX[(tradeHour >= 9) & (tradeHour <= 17)] # Fleming does 7:30 to 5. Doesn't matter for end-of-day obs.



