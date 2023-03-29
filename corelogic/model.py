import numpy as np
import pandas as pd

import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

munits.registry[np.datetime64] = mdates.ConciseDateConverter()

# Initial effective serviceability rate is an 0.1% cash rate, plus a 2.5 percentage
# point lender margin, plus a 3% APRA serviceability buffer:
r0 = 0.056

# Loan term in years
tau_loan = 30


tau = 200
alpha = 0.0
DATE_OF_FORECAST = np.datetime64('2022-08-09')

tau = 110
alpha = 10
DATE_OF_FORECAST = np.datetime64('2022-10-01')

CBA = False
INSANE = False
STANDARD = False
CURRENT = True

START_DATE = np.datetime64('2022-05-07')  # national
# START_DATE = np.datetime64('2022-08-27') # perth

# tau = 1
# alpha = 0


def get_index_data():
    df = pd.read_csv('data.csv')
    dates = np.array([np.datetime64(d) for d in df['Date']])[::-1]
    index = np.array(df['5 Cap City Aggregate (AUSD)'])[::-1]
    # index = np.array(df['Sydney(SYDD)'])[::-1]
    # index = np.array(df['Melbourne (MELD)'])[::-1]
    # index = np.array(df['Brisbane inc Gold Coast (BRID)'])[::-1]
    # index = np.array(df['Adelaide (ADED)'])[::-1]
    # index = np.array(df['Perth (PERD)'])[::-1]
    return dates, index


def model_index(r, I):
    # Given effective rate r (mortgage rate plus serviceability buffer) and nominal
    # income index I (where I=1 on May 7th), return modelled house price index (by
    # definition 1.0 on May 7th)
    P0 = 1 / r0 * (1 - (1 + r0) ** -tau_loan) + alpha
    P = I / r * (1 - (1 + r) ** -tau_loan) + alpha * I
    return (P / P0) * index_alltime_high
    # return (1 - (1 - alpha) * (1 - P / P0)) * index_alltime_high


def make_model(cashrates, wpirates, aprarates):
    t = np.arange(START_DATE, np.datetime64('2025-12-31'))
    cashrate_array = np.zeros(len(t))
    for date, cashrate in cashrates.items():
        date = np.datetime64(date)
        cashrate_array[t >= date] = cashrate / 100

    aprarate_array = np.zeros(len(t))
    for date, aprarate in aprarates.items():
        date = np.datetime64(date)
        aprarate_array[t >= date] = aprarate / 100

    r = cashrate_array + 0.025 + aprarate_array

    wpi_daily = np.zeros(len(t))
    for date, wpirate in wpirates.items():
        date = np.datetime64(date)
        wpi_daily[t >= date] = (1 + wpirate / 100) ** (4 / 365)

    I = wpi_daily.cumprod()

    index = model_index(r, I)
    index[0] = index_alltime_high
    return t, index


def ema(data, tau=200):
    alpha = 1 - np.exp(-1 / tau)
    smoothed_data = data.copy()
    for i, element in enumerate(smoothed_data[1:], start=1):
        smoothed_data[i] = alpha * element + (1 - alpha) * smoothed_data[i - 1]
    return smoothed_data


def wmr_model():
    t = np.arange(START_DATE, np.datetime64('2025-12-31'))
    dt = (np.datetime64('2025-12-31') - START_DATE).astype(int)
    tau = dt / np.log((145.40 / 2) / index_alltime_high)
    return t, index_alltime_high * np.exp((t.astype(int) - t.astype(int)[0]) / tau)


def percent_change(x):
    return 100 * (x / index_alltime_high - 1)


M = 28
N = 30


def m_day_average(x):
    ret = np.cumsum(x, dtype=float)
    return (ret[M:] - ret[:-M]) / M


def n_day_change(x, prepend=None):
    if prepend is not None:
        x = np.concatenate([prepend, x])
    return 100 * (x[N:] / x[:-N] - 1)


dates, index = get_index_data()
index_alltime_high = index[dates == START_DATE][0]  # 176.66

cashrates_ib = {
    '2022-01-01': 0.10,
    '2022-05-03': 0.35,
    '2022-06-07': 0.85,
    '2022-07-05': 1.35,
    '2022-08-02': 1.85,
    '2022-09-06': 2.263,  # interbank futures from here
    '2022-10-04': 2.611,
    '2022-11-01': 3.008,
    '2022-12-06': 3.284,
    '2023-01-03': 3.317,
    '2023-02-07': 3.491,
    '2023-03-07': 3.540,
    '2023-04-04': 3.506,
    '2023-05-09': 3.501,
    '2023-06-06': 3.487,
    '2023-07-04': 3.498,
    '2023-08-08': 3.494,
    '2023-09-05': 3.445,
    '2023-10-03': 3.411,
    '2023-11-07': 3.360,
    '2023-12-05': 3.316,
}

cashrates_insane = {
    '2022-01-01': 0.10,
    '2022-05-03': 0.35,
    '2022-06-07': 0.85,
    '2022-07-05': 1.35,
    '2022-08-02': 1.85,
    '2022-09-06': 2.300,  # interbank futures from here
    '2022-10-04': 2.692,
    '2022-11-01': 3.067,
    '2022-12-06': 3.357,
    '2023-01-03': 3.381,
    '2023-02-07': 3.626,
    '2023-03-07': 3.768,
    '2023-04-04': 3.864,
    '2023-05-09': 3.936,
    '2023-06-06': 3.985,
    '2023-07-04': 4.044,
    '2023-08-08': 4.050,
    '2023-09-05': 4.036,
    '2023-10-03': 4.001,
    '2023-11-07': 3.950,
    '2023-12-05': 3.900,
}

cashrates_cba = {
    '2022-01-01': 0.10,
    '2022-05-03': 0.35,
    '2022-06-07': 0.85,
    '2022-07-05': 1.35,
    '2022-08-02': 1.85,
    '2022-09-06': 2.35,  # cba forecast from here
    '2022-10-04': 2.60,
    '2022-11-01': 2.60,
    '2023-06-01': 2.35,
    '2023-09-01': 2.10,
    # '2022-11-01': 2.85,
    # '2022-12-06': 3.10,
    # '2023-02-07': 3.35,
    # '2023-03-07': 3.60,
    # '2023-10-01': 3.25,
    # '2023-12-01': 3.10,
}

aprarates_null = {'2022-01-01': 3.00}
aprarates_50bps = {
    '2022-01-01': 3.00,
    '2023-01-01': 2.50,
}
aprarates_100bps = {
    '2022-01-01': 3.00,
    '2023-01-01': 2.00,
}

# WPI reality so far:

# Jun 2021: 0.4
# Sep 2021 0.6
# Dec 2021: 0.65
# Mar 2022: 0.65
# Jun 2022: 0.70
# Sep 2022: 1.0

# RBA forecasts:

# Dec 2022: 3.0 YoY ⇒ 0.825 QoQ
# Jun 2023: 3.4 YoY ⇒ 0.825 QoQ
# Dec 2023: 3.6 YoY ⇒ 0.975 QoQ
# Jun 2024: 3.8 YoY ⇒ 0.975 QoQ
# Dec 2024: 3.9 YoY ⇒ 0.975 QoQ

# def infer_qoq(q1, q2, q3, yoy):
#     growth = (1 + yoy / 100) / ((1 + q1 / 100) * (1 + q2 / 100) * (1 + q3 / 100))
#     return 100 * (growth - 1)

# def infer_qoq2(q1, q2, yoy):
#     growth = ((1 + yoy / 100) / ((1 + q1 / 100) * (1 + q2 / 100))) ** (0.5)
#     return 100 * (growth - 1)


wpirates_rba = {
    '2022-01-01': 0.65,  # 22 Mar quarter
    '2022-04-01': 0.70,  # 22 Jun Quarter
    '2022-07-01': 1.100,  # 22 Sep quarter
    '2022-10-01': 0.800,  # 22 Dec quarter
    '2023-01-01': 0.969,  # RBA forecast from here. 23 Mar and Jun quarters
    '2023-06-01': 1.098,  # 23 Sep and Dec quarters
    '2024-01-01': 0.921,  # 24 Mar and Jun quarters
    '2024-06-01': 1.05,  # 24 Sep and Dec quarters
}


cashrates_actual = {
    '2022-01-01': 0.10,
    '2022-05-03': 0.35,
    '2022-06-07': 0.85,
    '2022-07-05': 1.35,
    '2022-08-02': 1.85,
    '2022-09-06': 2.35,
    '2022-10-04': 2.60,
    '2022-11-01': 2.85,
    '2022-12-06': 3.10,
    '2023-02-07': 3.35,
    '2023-03-07': 3.60,
}

aprarates_actual = {
    '2022-01-01': 3.00,
}

aprarates_100bps_mar = {
    '2022-01-01': 3.00,
    '2023-03-01': 2.00,
}

wpirates_actual = {
    '2022-01-01': 0.65,  # 22 Mar quarter
    '2022-04-01': 0.70,  # 22 Jun quarter
    '2022-07-01': 1.10,  # 22 Sep quarter
    '2022-10-01': 0.80,  # 22 Dec quarter
    '2023-01-01': 0.969,  # forecast
}


cashrates_385 = {
    '2022-01-01': 0.10,
    '2022-05-03': 0.35,
    '2022-06-07': 0.85,
    '2022-07-05': 1.35,
    '2022-08-02': 1.85,
    '2022-09-06': 2.35,
    '2022-10-04': 2.60,
    '2022-11-01': 2.85,
    '2022-12-06': 3.10,
    '2023-02-07': 3.35,
    '2023-03-07': 3.60,
    # '2023-04-04': 3.85,
    # '2023-05-02': 4.10,
}

model_dates, ib_model_index = make_model(cashrates_ib, wpirates_rba, aprarates_null)
model_dates, cba_model_index = make_model(cashrates_cba, wpirates_rba, aprarates_null)

model_dates, ib_apra50_model_index = make_model(
    cashrates_ib, wpirates_rba, aprarates_50bps
)
model_dates, cba_apra50_model_index = make_model(
    cashrates_cba, wpirates_rba, aprarates_50bps
)

model_dates, ib_apra100_model_index = make_model(
    cashrates_ib, wpirates_rba, aprarates_100bps
)
model_dates, cba_apra100_model_index = make_model(
    cashrates_cba, wpirates_rba, aprarates_100bps
)

model_dates, insane_index = make_model(cashrates_insane, wpirates_rba, aprarates_null)

model_dates, cashrate_385_index = make_model(
    cashrates_385, wpirates_rba | wpirates_actual, aprarates_null
)

# model_dates, cashrate_385_apra100_index = make_model(
#     cashrates_385, wpirates_rba | wpirates_actual, aprarates_100bps_mar
# )

model_dates, actual_index = make_model(
    cashrates_actual, wpirates_actual, aprarates_actual
)

wmr_dates, wmr_index = wmr_model()


plt.figure(figsize=(8, 7))

plt.plot(dates, percent_change(index), linewidth=3.5, color='k', label='Actual')
if STANDARD:
    plt.plot(
        model_dates,
        percent_change(ema(ib_model_index, tau=tau)),
        color='C0',
        linestyle='-',
        label='model (IB futures cash rate)',
    )
    plt.plot(
        model_dates,
        percent_change(ema(ib_apra50_model_index, tau=tau)),
        color='C0',
        linestyle='--',
        label='model (IB futures cash rate, APRA buffer -50bps)',
    )

    plt.plot(
        model_dates,
        percent_change(ema(ib_apra100_model_index, tau=tau)),
        color='C0',
        linestyle=':',
        label='model (IB futures cash rate, APRA buffer -100bps)',
    )
if CBA:
    plt.plot(
        model_dates,
        percent_change(ema(cba_model_index, tau=tau)),
        color='C1',
        linestyle='-',
        label='model (CBA forecast cash rate)',
    )
    plt.plot(
        model_dates,
        percent_change(ema(cba_apra50_model_index, tau=tau)),
        color='C1',
        linestyle='--',
        label='model (CBA cash rate, APRA buffer -50bps)',
    )
    plt.plot(
        model_dates,
        percent_change(ema(cba_apra100_model_index, tau=tau)),
        color='C1',
        linestyle=':',
        label='model (CBA cash rate, APRA buffer -100bps)',
    )
if INSANE:
    plt.plot(
        model_dates,
        percent_change(ema(insane_index, tau=tau)),
        color='C3',
        # linestyle=':',
        label='model (IB futures cash rate as of Aug 29th)',
    )

if CURRENT:
    plt.plot(
        model_dates,
        percent_change(ema(cashrate_385_index, tau=tau)),
        color='C4',
        linestyle=':',
        label='model (cashrate held at 3.6%)',
    )
    # plt.plot(
    #     model_dates,
    #     percent_change(ema(cashrate_385_apra100_index, tau=tau)),
    #     color='C3',
    #     linestyle=':',
    #     label='model (cashrate to 3.85% in 25bps increments, APRA -1% in March)',
    # )

plt.plot(
    model_dates[model_dates <= dates.max()],
    percent_change(ema(actual_index[model_dates <= dates.max()], tau=tau)),
    color='C4',
    linestyle='-',
    label='model (actual path of cash rate/APRA/wages)',
)

# plt.plot(
#     wmr_dates,
#     percent_change(wmr_index),
#     color='C2',
#     linestyle='-',
#     label='WMR prediction by EOY 2025 @ constant MoM decline',
# )

plt.axvline(
    DATE_OF_FORECAST,
    linestyle="--",
    color='k',
    label=f"date of forecast ({DATE_OF_FORECAST})",
)
plt.grid(True, color='k', linestyle="-", alpha=0.25)
plt.ylabel('Change from peak (%)')
plt.legend(loc='lower left', prop={'size': 9})
plt.axis(
    # xmin=dates.min(),
    xmin=np.datetime64('2021-06-01'),
    xmax=np.datetime64('2024-01-01'),
    ymin=-40,
    ymax=2.5,
)
plt.axhline(0, color='k')
plt.title(f'Property price forecast: serviceability model with {tau}-day lag')


plt.figure(figsize=(8, 7))

month_before_peak = index[(dates >= START_DATE - N) & (dates < START_DATE)]

plt.plot(
    dates[N:],
    n_day_change(index),
    linewidth=2.5,
    color='k',
    label='Actual',
)

if STANDARD:
    plt.plot(
        model_dates,
        n_day_change(ema(ib_model_index, tau=tau), prepend=month_before_peak),
        color='C0',
        linestyle='-',
        label='model (IB futures cash rate)',
    )
    plt.plot(
        model_dates,
        n_day_change(ema(ib_apra50_model_index, tau=tau), prepend=month_before_peak),
        color='C0',
        linestyle='--',
        label='model (IB futures cash rate, APRA buffer -50bps)',
    )

    plt.plot(
        model_dates,
        n_day_change(ema(ib_apra100_model_index, tau=tau), prepend=month_before_peak),
        color='C0',
        linestyle=':',
        label='model (IB futures cash rate, APRA buffer -100bps)',
    )

if CBA:
    plt.plot(
        model_dates,
        n_day_change(ema(cba_model_index, tau=tau), prepend=month_before_peak),
        color='C1',
        linestyle='-',
        label='model (CBA forecast cash rate)',
    )
    plt.plot(
        model_dates,
        n_day_change(ema(cba_apra50_model_index, tau=tau), prepend=month_before_peak),
        color='C1',
        linestyle='--',
        label='model (CBA cash rate, APRA buffer -50bps)',
    )
    plt.plot(
        model_dates,
        n_day_change(ema(cba_apra100_model_index, tau=tau), prepend=month_before_peak),
        color='C1',
        linestyle=':',
        label='model (CBA cash rate, APRA buffer -100bps)',
    )

if INSANE:
    plt.plot(
        model_dates,
        n_day_change(ema(insane_index, tau=tau), prepend=month_before_peak),
        color='C3',
        # linestyle=':',
        label='model (IB futures cash rate as of Aug 29th)',
    )

if CURRENT:
    plt.plot(
        model_dates,
        n_day_change(ema(cashrate_385_index, tau=tau), prepend=month_before_peak),
        color='C4',
        linestyle=':',
        label='model (cashrate to 3.85% in 25bps increments)',
    )
    # plt.plot(
    #     model_dates,
    #     n_day_change(ema(cashrate_385_apra100_index, tau=tau), prepend=month_before_peak),
    #     color='C3',
    #     linestyle=':',
    #     label='model (cashrate to 3.85% in 25bps increments, APRA -1% in March)',
    # )

plt.plot(
    model_dates[model_dates <= dates.max()],
    n_day_change(
        ema(actual_index[model_dates <= dates.max()], tau=tau),
        prepend=month_before_peak,
    ),
    color='C4',
    linestyle='-',
    label='model (actual path of cash rate/APRA/wages)',
)

# plt.plot(
#     wmr_dates,
#     n_day_change(wmr_index, prepend=month_before_peak),
#     color='C2',
#     linestyle='-',
#     label='WMR prediction by EOY 2025 @ constant MoM decline',
# )

plt.axvline(
    DATE_OF_FORECAST,
    linestyle="--",
    color='k',
    label=f"date of forecast ({DATE_OF_FORECAST})",
)
plt.grid(True, color='k', linestyle="-", alpha=0.25)
plt.ylabel('30-day change (%)')
plt.legend(loc='upper right', prop={'size': 9})
plt.axis(
    # xmin=dates.min() + 30,
    xmin=np.datetime64('2021-06-01'),
    xmax=np.datetime64('2024-01-01'),
    ymin=-2.5,
    ymax=3.0,
)
plt.axhline(0, color='k')
plt.title(f'Property price forecast: serviceability model with {tau}-day lag')

plt.show()
