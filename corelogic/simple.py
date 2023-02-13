# Thanks to /u/shrugmeh for the data

# Bears who think the model is crap and that drops will be bigger than it implies:
#
# /u/Reclusiarc
# /u/HugeCanoe
#
# /u/LelcoinDegen says 0.5 0.25, then cut in March 2023
# https://www.reddit.com/r/AusFinance/comments/wyw63j/comment/ilzazwy/
#


import numpy as np
import pandas as pd

import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

munits.registry[np.datetime64] = mdates.ConciseDateConverter()


def get_index_data():
    df = pd.read_csv('data.csv')
    dates = np.array([np.datetime64(d) for d in df['Date']])[::-1]
    index = np.array(df['5 Cap City Aggregate (AUSD)'])[::-1]
    # index = np.array(df['Sydney(SYDD)'])[::-1]
    # index = np.array(df['Melbourne (MELD)'])[::-1]
    # index = np.array(df['Brisbane inc Gold Coast (BRID)'])[::-1]
    # index = np.array(df['Adelaide (ADED)'])[::-1]
    # index = np.array(df['Perth (PERD)'])[::-1]

    # for _ in range(90):
    #     dates = np.append(dates, [dates[-1] + 1])
    #     index = np.append(index, [index[-1]*(1-0.01475)**(1/30)])
    return dates, index


def const_accel_model(start_index, start_date, velocity_factor):
    t = np.arange(start_date, np.datetime64('2025-12-31'))
    daily_accel = (
        velocity_factor[30:] ** (12 / 365) - velocity_factor[:-30] ** (12 / 365)
    )[-1] / 30
    daily_velocity_factor = velocity_factor[-1] ** (12 / 365)
    modelled = [start_index]
    modelled_velocity = [daily_velocity_factor]
    for _ in t[1:]:
        modelled.append(modelled_velocity[-1] * modelled[-1])
        modelled_velocity.append(modelled_velocity[-1] + daily_accel)
    for _ in range(60):
        t = np.append([t[0] - 1], t)
        modelled.insert(0, modelled[0] / modelled_velocity[0])
        modelled_velocity.insert(0, modelled_velocity[0] - daily_accel)
    return t, np.array(modelled)


def quadratic_model(dates, index):

    NPTS = 120

    t = np.arange(dates[-1] - NPTS, np.datetime64('2025-12-31'))
    t_fit = (dates[-NPTS:] - dates[-1]).astype(float)
    from scipy.optimize import curve_fit

    def quadratic(t, A, t0, c):
        return A * (t - t0) ** 2 + c

    params, cov = curve_fit(quadratic, t_fit, index[-NPTS:])

    modelled = quadratic((t - dates[-1]).astype(float), *params)

    return t, modelled


def percent_change(x):
    return 100 * (x / index_alltime_high - 1)


dates, index = get_index_data()

index_alltime_high = index[dates == np.datetime64('2022-05-07')][0]  # 176.66
index_alltime_high = index.max()


N = 28

velocity_factor = (index[30:] / index[:-30]) ** (365 / 12 / 30)

const_accel_dates, const_accel_index = const_accel_model(
    index[-1], dates[-1], velocity_factor
)

plt.figure(figsize=(8, 7))

mindates = []
minvals = []
latest_velocity = 100 * (velocity_factor[-1] - 1)
latest_accel = 100 * (velocity_factor[-1] - velocity_factor[-31])

for i in range(-20, 0):
    # for i in range(-20, 0):
    const_accel_dates, const_accel_index = const_accel_model(
        index[i], dates[i], velocity_factor[:i]
    )
    # const_accel_dates, const_accel_index = quadratic_model(dates[:i], index[:i])
    plt.plot(
        const_accel_dates,
        percent_change(const_accel_index),
        linewidth=2,
        # linestyle=":",
        color='C4',
        alpha=0.5 if i == -1 else 0.1,
        label=f"Constant acceleration trendline"
        # + " ({latest_velocity:+.2f}%/m, {latest_accel:+.2f}%/m/m)"
        if i == -1 else '',
    )
    mindates.append(const_accel_dates[const_accel_index.argmin()])
    minvals.append(const_accel_index.min())


plt.plot(
    dates,
    percent_change(index),
    linewidth=3.5,
    color='k',
    label=f"Index ({percent_change(index)[-1]:.1f}% from peak)",
)

plt.plot(
    mindates,
    percent_change(minvals),
    'ko-',
    label='Previous extrapolated bottom',
    alpha=0.25,
    markersize=3,
    markeredgewidth=0.0,
)
plt.plot(
    mindates[-1],
    percent_change(minvals)[-1],
    'ro',
    label=f"Latest extrapolated bottom ({percent_change(minvals)[-1]:.1f}% from peak on {mindates[-1]})",
)

plt.grid(True, color='k', linestyle="-", alpha=0.25)
plt.ylabel('Change from peak (%)')
plt.axis(
    # xmin=dates.min(),
    xmin=np.datetime64('2019-06-01'),
    xmax=np.datetime64('2024-01-01'),
    ymin=-40,
    ymax=2.5,
)
plt.axhline(0, color='k')
plt.title('CoreLogic 5-capitals index change from peak')
plt.legend()


# plt.figure(figsize=(8, 7))


# const_accel_dates, const_accel_index = const_accel_model(
#         index[-1], dates[-1], velocity_factor[:-1]
#     )
# # const_accel_dates, const_accel_index = quadratic_model(dates[:-1], index[:-1])

# plt.plot(
#     dates[-60:],
#     percent_change(index)[-60:] - percent_change(const_accel_index)[2:62],
#     linewidth=3.5,
#     color='k',
#     label="Diff",
# )

# plt.grid(True, color='k', linestyle="-", alpha=0.25)
# plt.ylabel('Change from peak (%)')
# plt.axis(
#     # xmin=dates.min(),
#     xmin=np.datetime64('2019-06-01'),
#     xmax=np.datetime64('2024-01-01'),
#     ymin=-40,
#     ymax=2.5,
# )
# plt.axhline(0, color='k')
# plt.title('Difference from quadratic model')
# plt.legend()


# plt.figure(figsize=(8, 7))
# plt.plot(mindates, percent_change(minvals), 'ko', label='previous extrapolations')
# plt.plot(mindates[-1], percent_change(minvals)[-1], 'ro', label='latest extrapolation')
# plt.grid(True, color='k', linestyle="-", alpha=0.25)
# plt.ylabel('Extrapolated peak to trough drop (%)')
# plt.xlabel('Extrapolated date of market bottom')
# plt.title('Trough depth and date on constant acceleration trend')
# plt.legend()
# plt.axis(
#     # xmin=dates.min(),
#     xmin=np.datetime64('2019-06-01'),
#     xmax=np.datetime64('2024-01-01'),
#     ymin=-40,
#     ymax=2.5,
# )
# plt.axhline(0, color='k')

plt.show()
