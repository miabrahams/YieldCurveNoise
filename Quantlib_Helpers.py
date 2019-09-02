import pandas as pd
import QuantLib2 as ql
import numpy as np
import math

def timestamp_to_qldate(ts):
    def _timestamp_to_qldate(t):
        t = pd.Timestamp(t)
        return ql.Date(t.day, t.month, t.year)
    try:
        return [_timestamp_to_qldate(t) for t in ts]
    except TypeError:
        return _timestamp_to_qldate(ts)

def annual_to_continuous(y):
    return math.ln(1+y)

def ql_array(iterable):
    A = ql.Array(len(iterable))
    for i,b in enumerate(iterable):
        A[i] = float(b)
    return A


# Nelson, Svensson, Siegel yield curve using Quantlib Kappa substitution
_maturities = np.arange(0.0, 121.0/12.0, 1.0/12.0)
_maturities[0] = .001
def nss_yield(params):
    [Beta0, Beta1, Beta2, Beta3, Kappa1, Kappa2] = [p for p in params]
    return  [Beta0 + Beta1 * (1 - np.exp(-n * Kappa1)) / (n * Kappa1) + \
             Beta2 * ((1 - np.exp(-n * Kappa1))/(n * Kappa1) - np.exp(-n * Kappa1)) + \
             Beta3 * ((1 - np.exp(-n * Kappa2))/(n * Kappa2) - np.exp(-n * Kappa2)) for n in _maturities]


def par_yield_semiannual(yc, mat_date, debug=False):
    dc = ql.SimpleDayCounter()  # We use a simple day counter for these theoretical rate constructions.
    calc_date = yc.referenceDate()
    if dc.yearFraction(calc_date, mat_date) < 1/30:
        return yc.zeroRate(mat_date, yc.dayCounter(), ql.Annual).rate()
    schedule = ql.Schedule(calc_date,
                           mat_date,
                           ql.Period(6, ql.Months),
                           ql.NullCalendar(),
                           ql.Unadjusted,
                           ql.Unadjusted,
                           ql.DateGeneration.Backward, False)
    discounts = np.array([yc.discount(d) for d in schedule])
    if schedule[0] != calc_date:  # schedule[0] is just today so we won't use it
        print("Schedule[0] wasn't today!")
    first_coupon_date = schedule[1]
    discounts = discounts[1:]
    accrfrac = (1 - dc.yearFraction(calc_date, first_coupon_date) * 2.0)
    par_yield = 2 * (1 - discounts[-1]) / (np.sum(discounts) - accrfrac)
    if debug:
        print(discounts)
        print(accrfrac)
    return par_yield


class QuantlibFitOptions:
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    business_convention = ql.ModifiedFollowing  # Should this be ql.Unadjusted or ModifiedFollowing???
    day_counter = ql.ActualActual()
    end_of_month = True
    settlement_days = 0
    face_amount = 100
    coupon_frequency = ql.Semiannual
    compounding = ql.Compounded
