from mga_helpers import *
import numpy as np

# Look into this! Kinda cute lol
# import fredapi as fred


HPW_Noise = pd.read_excel('data/HPW/HPW_Noise_Measure.xlsx')
HPW_Noise = HPW_Noise.set_index('Date')
HPW_Noise.to_pickle('data/HPW/HPW.pickle')





# Fama-French portfolios
# FF data are not annualized.
# Date like 201801 means return from Jan 1 to Jan 31 in 2018
# First obs is 192607 so output should be 1926-07-31
FF:pd.DataFrame = pd.read_csv('data/FamaFrench/F-F_Research_Data_Factors.csv')
FF.date = pd.to_datetime(FF.date.astype(str), format="%Y%m")
FF = FF.set_index('date')
FF = make_monthly(FF, 'bfill')
FF = FF / 100.0
FF = FF.rename(columns = {'Mkt-RF': "MktRx"})
FF.to_pickle('data/FamaFrench/FF3Facs.pickle')

# Currency portfolios
FXReturns = pd.read_csv('data/FX/FXPortfolios.csv', index_col=[0], parse_dates=[0])
for col in FXReturns.columns:
    FXReturns[col] = pd.to_numeric(FXReturns[col].str[:-1]) / 100.00
FXReturns = make_monthly(FXReturns, 'bfill')
FXReturns.to_pickle('data/FX/FXPortfolios.pklz', compression='gzip')





# Big Correlation Table
TreasuriesLiborFRED = pd.read_csv("data/BigCorrelationTable/Treasuries_LIBOR_Fred.csv", index_col=0, parse_dates=True, na_values=".") # 10year, 1year, 5year, 3m, 3m LIBOR
TreasuriesLiborFRED.dropna(inplace=True)
TreasuriesLiborFRED['BondVol'] = TreasuriesLiborFRED['DGS5'].rolling(21).std() * np.sqrt(12)
# TreasuriesLiborFRED.index = pd.DatetimeIndex(TreasuriesLiborFRED['DATE'])
# TrasuriesLiborFRED = TreasuriesLiborFRED.drop('DATE', axis=1)
TreasuriesLiborFRED = make_monthly(TreasuriesLiborFRED)


CorporateSpreads = pd.read_csv("data/BigCorrelationTable/BaaAaaSpread_FRED.csv", index_col=0, parse_dates=True) # Baa Aaa
CorporateSpreads = make_monthly(CorporateSpreads)

GCRepo = pd.read_excel("data/BigCorrelationTable/HistoricalOvernightTreasGCRepoPriDealerSurvRate.xlsx", index_col=0, na_values=".", parse_dates=True) # GC Repo
GCRepo = pd.to_numeric(GCRepo.iloc[:,0], errors='coerce')
GCRepo.dropna(inplace=True)
GCRepo = make_monthly(GCRepo)

MKTReturn = pd.read_csv("data/BigCorrelationTable/MonthlyMKTReturn_CRSP.csv", index_col=0, parse_dates=True) # vwretd
MKTReturn = make_monthly(MKTReturn)

VIXFRED = pd.read_csv("data/BigCorrelationTable/VIX_Fred.csv", index_col=0, parse_dates=True, na_values=".") # VIX
VIXFRED  = make_monthly(VIXFRED)

PastorStambauch = pd.read_csv("data/BigCorrelationTable/PastorStambauch.csv", index_col=0, parse_dates=True, date_parser=lambda d: pd.to_datetime(d, format="%Y%m")) # PS
# PastorStambauch.rename(columns={'Agg Liq.': "PS"}, inplace=True)
PastorStambauch.rename(columns={'Innov Liq (eq8)': "PS"}, inplace=True)
PastorStambauch['PS'] = PastorStambauch['PS'].where(lambda x: x > -98.0, other=np.nan)
PastorStambauch = make_monthly(PastorStambauch)


# merge_params = {"left_index" :True, "right_index" : True, "how" :"outer"}
merge_params = { "how" :"outer"}

BigTableDataframe = TreasuriesLiborFRED.USD3MTD156N.to_frame().rename(columns = {"USD3MTD156N": 'treas3m'})
BigTableDataframe = BigTableDataframe.join(TreasuriesLiborFRED['DTB3'].rename("Libor").to_frame(), how="outer")
BigTableDataframe = BigTableDataframe.join(VIXFRED['VIXCLS'].rename('VIX').to_frame(), **merge_params)
BigTableDataframe = BigTableDataframe.join(PastorStambauch['PS'].to_frame(), **merge_params)
BigTableDataframe = BigTableDataframe.join(MKTReturn['vwretd'].rename('ValueWeightedMKT').to_frame(), **merge_params)
BigTableDataframe = BigTableDataframe.join((CorporateSpreads['BAA'] - CorporateSpreads['AAA']).rename('Baa_Aaa').to_frame(), **merge_params)
BigTableDataframe = BigTableDataframe.join((TreasuriesLiborFRED['DGS10'] - TreasuriesLiborFRED['DGS1']).rename('slope_10y').to_frame(), **merge_params)
BigTableDataframe = BigTableDataframe.join(TreasuriesLiborFRED['BondVol'].to_frame(), **merge_params)
BigTableDataframe = BigTableDataframe.join(GCRepo.rename('GCRepo').to_frame(), **merge_params)

BigTableDataframe.count()
BigTableDataframe.dtypes
BigTableDataframe


BigTableDataframe.dropna(thresh=2, inplace=True)

BigTableDataframe.to_pickle("data/BigCorrelationTable/BigTableDataframe.pklz", compression='gzip')


# TODO: On-The-Run Spread, RefCorp, GC repo rate before 1998

