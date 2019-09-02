# Important stuff
import pandas as pd

# WHAT is going on here??
bad_CUSIPs = ['912810BF',
              '912810BG',
              '912810DV',  # Weird quotes in GovPX
              '912828GL',
              '912795AP' # One crazy day on 1998-07-10
              ]


spike_dates = pd.Series(pd.to_datetime(['1995-04-05', '1998-01-23', '1998-07-10', '1998-07-21', '1998-10-07', '1999-01-28', '2006-06-30', '2011-08-01', '2011-08-03', '2011-08-05']))


bad_dates = pd.Series(['1998-01-23', # Wow that one is crazy!
                      ])
bad_dates = pd.to_datetime(bad_dates).as_matrix()


def clean_bad_bonds(df_export):
    not_inflation_indexed = (df_export.ITYPE != 12) & (df_export.ITYPE != 11)
    not_callable = (df_export.ITYPE != 5) & (df_export.ITYPE != 6)
    not_excluded = ~(df_export.TCUSIP.isin(bad_CUSIPs))
    return df_export[not_inflation_indexed & not_callable & not_excluded].copy(deep=True)


def get_df_inplace(CRSPData, query_date):
    pd.to_datetime(query_date)
    selected_date = CRSPData.CALDT == pd.to_datetime(query_date)
    # Create data for export
    df_export = CRSPData[selected_date]
    df_export = clean_bad_bonds(df_export)
    print("Observations: " + str(df_export.shape[0]))
    return df_export


def get_df_inflation(CRSPData, query_date):
    pd.to_datetime(query_date)
    selected_date = CRSPData.CALDT == pd.to_datetime(query_date)
    # Create data for export
    df_export = CRSPData[selected_date]
    df_export = df_export.ix[(df_export.ITYPE == 12) | (df_export.ITYPE == 11)]
    print("Observations: " + str(df_export.shape[0]))
    return df_export
