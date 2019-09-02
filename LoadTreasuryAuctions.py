
from mga_helpers import *

import tqdm
import sqlite3

# Load info from CRSP dataset
unique_dates = pd.read_pickle("data/CRSPUniqueDates.pkl.compress", compression="gzip")
df_export = pd.read_pickle("data/daily_cross_section.pkl.compress", compression="gzip")


# Import Treasury Auction data to identify on-the-runs and first-off-the-runs by concatenating multiple CSVs
TreasuryAuctions = []
for i in range(1, 8):
    TreasuryAuctions = TreasuryAuctions + [pd.read_csv("data/TreasuryAuctions/Securities (" + str(i) + ").csv")]
TreasuryAuctions = pd.concat(TreasuryAuctions, ignore_index=True)
TreasuryAuctions = TreasuryAuctions.rename(index=str,
                                           columns={col: col.replace(" ", "") for col in TreasuryAuctions.columns})

TreasuryAuctions['CUSIP'] = TreasuryAuctions['CUSIP'].str[0:8] # remove checksum

# Clean
make_date_column(TreasuryAuctions, 'AuctionDate', fmt="%m/%d/%Y")
make_date_column(TreasuryAuctions, 'IssueDate', fmt="%m/%d/%Y")
make_date_column(TreasuryAuctions, 'MaturityDate', fmt="%m/%d/%Y")

DayBeforeSample = unique_dates[0] - pd.Timedelta(days=1) # Start just before the sample begins
DayBeforeSample = pd.Timestamp(1991, 1, 1)


IssueMaturities = {'_4W' : [28, 4],
                   '_13W': [91, 4],
                   '_26W': [182, 7],
                   '_52W': [364, 7],
                   '_2Y' : [730, 31],
                   '_3Y' : [1095, 31],
                   '_5Y' : [1825, 31],
                   '_7Y' : [2555, 31],
                   '_10Y': [3650, 31],
                   '_20Y': [7300, 31],
                   '_30Y': [10950, 62]}


OnTheRunExportCols = ['Date', 'Status'] + list(IssueMaturities.keys())
CurrentOnTheRun = dict(zip(OnTheRunExportCols, [[DayBeforeSample], ['Current']] + [[None]] * 11))
LastOnTheRun = CurrentOnTheRun.copy()
LastOnTheRun['Status'] = 'Last'
SecondLastOnTheRun = LastOnTheRun.copy()
SecondLastOnTheRun['Status'] = 'SecondLast'
OnTheRunExport = pd.DataFrame(columns=OnTheRunExportCols)

CUSIPFrequency = TreasuryAuctions.CUSIP.value_counts() # Just for debugging
CurrentOnTheRunMatDate = dict(zip(IssueMaturities.keys(), [''] * 10)) # For sanity checking
CurrentDate = DayBeforeSample # For inner iteration
SeenCUSIPs = set()
unique_dates_pos = 0 # We're going to match every item in unique_dates


it = TreasuryAuctions.iloc[::-1].itertuples()
for i in tqdm(range(TreasuryAuctions.shape[0])):
    r = next(it)
    t = (r.MaturityDate - r.IssueDate).days
    foundMatch = False
    isRepeatedCusip = False
    if r.CUSIP in SeenCUSIPs:
        isRepeatedCusip = True
        print(" Seeing CUSIP " + r.CUSIP + " multiple times. New term is " + r.SecurityTerm)
    SeenCUSIPs.add(r.CUSIP)

    # Multiple auctions are held on the same day. So we will accumulate changes to CurrentOnTheRun over a given day,
    # then when we see an auction on a new day we perform this commit.
    # We miss the last set of auctions but since the Treasury auction data bookends the CRSP data we're fine.
    if CurrentDate < r.AuctionDate:
        CurrentDate = r.AuctionDate
        # print("Starting to do the append!")
        # Commit list of current on the run securities to the export matrix
        while unique_dates_pos < unique_dates.size and unique_dates[unique_dates_pos] <= r.AuctionDate:
            CurrentOnTheRun['Date'] = LastOnTheRun['Date'] = SecondLastOnTheRun['Date'] = [unique_dates_copy[0]]
            OnTheRunExport = OnTheRunExport.append(pd.DataFrame.from_dict(CurrentOnTheRun), ignore_index=True). \
                append(pd.DataFrame.from_dict(LastOnTheRun), ignore_index=True). \
                append(pd.DataFrame.from_dict(SecondLastOnTheRun), ignore_index=True)
            unique_dates_pos += 1

    for key, val in IssueMaturities.items():
        # Try to match a maturity date in buckets
        if (val[0] - val[1] < t) & (val[0] + val[1] > t):
            if isRepeatedCusip and CurrentOnTheRun[key] is r.CUSIP:
                print("Recent reissue of " + r.SecurityTerm + " security detected")
            else:
                SecondLastOnTheRun[key] = LastOnTheRun[key]
                LastOnTheRun[key] = CurrentOnTheRun[key]
                CurrentOnTheRun[key] = [r.CUSIP]
            foundMatch = True

    if foundMatch is False:
        if isRepeatedCusip:
            print("Seasoned reissue detected.")
        else:
            print("Could not match CUSIP " + r.CUSIP + " on " + nicedate(r.AuctionDate) +
                  ". Term: " + r.SecurityTerm + ". Maturity was " + str(t) + " days. ")


# Save pkl
OnTheRunExport.to_pickle("data/CurrentOnTheRun.pkl.compress", compression="gzip")
OnTheRunExport = pd.read_pickle("data/CurrentOnTheRun.pkl.compress", compression="gzip")

# Export SQL database
export_conn = sqlite3.connect('data/current_on_the_run')
OnTheRunExport.to_sql('current_on_the_run', export_conn, if_exists='replace')
export_conn.close()

# Drop duplicates?
first_cusip_issue = TreasuryAuctions.drop_duplicates(subset='CUSIP', keep='last')
first_cusip_issue = first_cusip_issue.set_index('CUSIP')
first_cusip_issue.to_pickle("data/first_issue_date.pkl.compress", compression="gzip")

