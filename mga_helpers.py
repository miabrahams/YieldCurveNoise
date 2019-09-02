import pandas as pd
from pandas.tseries.offsets import MonthEnd, Day

# global_fig = None
# global_axis = None

# def pyplot_deleted_callback(x):

def str2bool(v):
    import argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# It's important to specify the format when converting a Pandas column to datetime
def make_date_column(df, col, fmt="%m/%d/%Y"):
    dt = pd.to_datetime(df[col], format=fmt)
    df[col] = dt


def make_date_column_infer(df, col):
    dt = pd.to_datetime(df[col], infer_datetime_format=True)
    df[col] = dt


def make_monthly(df: pd.DataFrame, method='ffill', add_endpoints: list = None) -> pd.DataFrame:
    if add_endpoints:
        ind = df.index.to_series()
        if add_endpoints[0]:
            ind[0] = ind[0] - MonthEnd(1)
        if add_endpoints[1]:
            ind[-1] = ind[-1] + MonthEnd(1)
        df.set_index(ind, inplace=True)
    df = df.asfreq(freq='1M', method=method, how="e")
    df.index += MonthEnd(0)
    return df


# Make a pretty print string from np.datetime64
def nicedate(t):
    ts = pd.to_datetime(t)
    return f"{ts.month}/{ts.day}/{ts.year}"


def chunk(L, n):
    """Divide range(L) into n equally sized parts"""
    assert 0 < n <= L
    s, r = divmod(L, n)
    t = s + 1
    slices = [slice(p, p+t) for p in range(0, r*t, t)] + [slice(p, p+s) for p in range(r*t, L, s)]
    return slices


def quickplot(y, x=None, new=True):
    import matplotlib.pyplot as plt
    fig = plt.figure(7777)
    if new:
        fig.clf()
        fig, ax = plt.subplots(num=7777)
    else:
        ax = fig.gca()
    # global global_fig, global_axis
    # if global_fig is None:
    #     global_fig, global_axis = plt.subplots()
    if x is None:
        ax.plot(y)
    else:
        ax.plot(x, y)
    fig.show()

def qp(y, x=None):
    quickplot(y, x, True)

def qpadd(y, x=None):
    quickplot(y, x, False)


def savepklz(obj, filename):
    """
Save zipped pickle file.
    :param obj:  Any object
    :param filename:  Filename. Probably should end in .pklz
    """
    import gzip
    import pickle
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f)



def loadpklz(filename):
    """
Load zipped pickle file.
    :param filename:
    :return: Contents of pickle file.
    """
    import gzip
    import pickle
    with gzip.open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

