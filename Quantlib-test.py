# The rest of the points are coupon bonds. We assume that the YTM given for the bonds are all par rates.
# So we have bonds with coupon rate same as the YTM.

import QuantLib as ql


def print_curve(xlist, ylist, precision=3):
    """
    Method to print curve in a nice format
    """
    print("----------------------")
    print("Maturities\tCurve")
    print("----------------------")
    for x,y in zip(xlist, ylist):
        print(x,"\t\t", round(y, precision))
    print("----------------------")


# Deposit rates
depo_maturities = [ql.Period(6, ql.Months), ql.Period(12, ql.Months)]
depo_rates = [5.25, 5.5]

# Bond rates
bond_maturities = [ql.Period(6*i, ql.Months) for i in range(3, 21)]
bond_rates = [5.75, 6.0, 6.25, 6.5, 6.75, 6.80, 7.00, 7.1, 7.15,
              7.2, 7.3, 7.35, 7.4, 7.5, 7.6, 7.6, 7.7, 7.8]

print_curve(depo_maturities+bond_maturities, depo_rates+bond_rates)


# Lets define some of the constants required for the rest of the objects needed below.


# some constants and conventions
# here we just assume for the sake of example
# that some of the constants are the same for
# depo rates and bond rates

calc_date = ql.Date(15, 1, 2015)
ql.Settings.instance().evaluationDate = calc_date

calendar = ql.UnitedStates()
business_convention = ql.Unadjusted
day_count = ql.Thirty360()
end_of_month = True
settlement_days = 0
face_amount = 100
coupon_frequency = ql.Period(ql.Semiannual)
settlement_days = 0


# The basic idea of bootstrapping using QuantLib is to use the deposit rates and bond rates to create
# individual helpers. Then use the combination of the two helpers to construct the yield curve.


# create deposit rate helpers from depo_rates
depo_helpers = [ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(r/100.0)),
                                     m,
                                     settlement_days,
                                     calendar,
                                     business_convention,
                                     end_of_month,
                                     day_count )
                for r, m in zip(depo_rates, depo_maturities)]


schedule = ql.Schedule(calc_date,
                       calc_date + bond_maturities[0],
                       coupon_frequency,
                       calendar,
                       business_convention,
                       business_convention,
                       ql.DateGeneration.Backward,
                       end_of_month)


# create fixed rate bond helpers from fixed rate bonds
bond_helpers = []
for r, m in zip(bond_rates, bond_maturities):
    print(type(r))
    print(type(m))

    termination_date = calc_date + m
    schedule = ql.Schedule(calc_date,
                           termination_date,
                           coupon_frequency,
                           calendar,
                           business_convention,
                           business_convention,
                           ql.DateGeneration.Backward,
                           end_of_month)

    bond_helper = ql.FixedRateBondHelper(ql.QuoteHandle(ql.SimpleQuote(face_amount)),
                                         settlement_days,
                                         face_amount,
                                         schedule,
                                         [r / 100.0],
                                         day_count,
                                         business_convention,
                                         )
    bond_helpers.append(bond_helper)

# The yield curve is constructed here
rate_helpers = depo_helpers + bond_helpers
yieldcurve = ql.PiecewiseLogCubicDiscount(calc_date,
                                          rate_helpers,
                                          day_count)
# get spot rates
spots = []
tenors = []
for d in yieldcurve.dates():
    yrs = day_count.yearFraction(calc_date, d)
    compounding = ql.Compounded
    freq = ql.Semiannual
    zero_rate = yieldcurve.zeroRate(yrs, compounding, freq)
    tenors.append(yrs)
    eq_rate = zero_rate.equivalentRate(day_count,
                                       compounding,
                                       freq,
                                       calc_date,
                                       d).rate()
    spots.append(100 * eq_rate)
print_curve(tenors, spots)




# Pricing a fixed coupon bond


issueDate = ql.Date(15, 1, 2015)
maturityDate = ql.Date(15, 1, 2016)
tenor = ql.Period(ql.Semiannual)
calendar = ql.UnitedStates()
businessConvention = ql.Unadjusted
dateGeneration = ql.DateGeneration.Backward
monthEnd = False
schedule = ql.Schedule (issueDate, maturityDate, tenor, calendar, businessConvention, businessConvention , dateGeneration, monthEnd)
list(schedule)






# Now lets build the coupon
dayCount = ql.Thirty360()
couponRate = .06
coupons = [couponRate]

# Now lets construct the FixedRateBond
settlementDays = 0
faceValue = 100
fixedRateBond = ql.FixedRateBond(settlementDays, faceValue, schedule, coupons, dayCount)

bondEngine = ql.DiscountingBondEngine(spotCurveHandle)
fixedRateBond.setPricingEngine(bondEngine)

# Finally the price
fixedRateBond.NPV()
# >>> 105.27653992490683

