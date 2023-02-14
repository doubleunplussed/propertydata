import numpy as np
import pandas as pd

import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

munits.registry[np.datetime64] = mdates.ConciseDateConverter()




def get_index_data():
    df = pd.read_csv('data.csv')
    dates = np.array([np.datetime64(d) for d in df['Date']])[::-1]
    return dates, {
        '5 capital city aggregate': np.array(df['5 Cap City Aggregate (AUSD)'])[::-1],
        'Sydney': np.array(df['Sydney(SYDD)'])[::-1],
        'Melbourne': np.array(df['Melbourne (MELD)'])[::-1],
        'Brisbane': np.array(df['Brisbane inc Gold Coast (BRID)'])[::-1],
        'Adelaide': np.array(df['Adelaide (ADED)'])[::-1],
        'Perth': np.array(df['Perth (PERD)'])[::-1],
        '5-city aggregate': np.array(df['5 Cap City Aggregate (AUSD)'])[::-1],
    }

N = 30


def n_day_average(data, n):
    ret = np.cumsum(data, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

def n_day_change(x, prepend=None):
    if prepend is not None:
        x = np.concatenate([prepend, x])
    return 100 * (x[N:] / x[:-N] - 1)


all_dates, data = get_index_data()

# all_dates = all_dates[:-30]
# for CITY in data:
#     data[CITY] = data[CITY][:-30]

plt.figure()
CITY = '5-city aggregate'
DROP = 10

CITY = 'Sydney'
DROP = 40

SMOOTHED = False
SMOOTHED = True

index = data[CITY]

smoothed_index = n_day_average(index, 28)

peak_date = all_dates[index.argmax()]
TARGET = (1 - DROP / 100) * index.max()

peak_index = index[all_dates.searchsorted(peak_date)]

if SMOOTHED:
    velocity = n_day_change(smoothed_index)[all_dates[N:] > peak_date + N]
else:
    velocity = n_day_change(index)[all_dates[N:] > peak_date + N]


index = index[all_dates > peak_date + N]
dates = all_dates[all_dates > peak_date + N]

# plt.plot(dates, velocity)
# plt.show()



k = (1 + velocity / 100) ** (1 / N) - 1
t = dates + np.ceil(1 / k * np.log(TARGET / index)).astype(int)
# t = 1 / k * np.log(EOY_2020_index / index)

plt.plot(
    dates,
    t,
    label=("Date of 40% drop" if CITY == "Sydney" else "Hat time")
    + f" | {CITY} ({t[-1]})",
)

plt.plot(
    dates,
    dates,
    'k--',
    label="Current date" if CITY == "Sydney" else "Now time",
)

if SMOOTHED:
    plt.title(f"Projected date of {DROP}% loss @ rolling (smoothed) MoM decline")
else:
    plt.title(f"Projected date of {DROP}% loss @ rolling MoM decline")

plt.ylabel(f"Projected date of {DROP}% drop")

if CITY == 'Sydney':
    plt.axis(
        xmin=np.datetime64('2022-06-01'),
        ymin=np.datetime64('2023-01-01'),
        ymax=np.datetime64('2028-01-01'),
    )
else:
    plt.axis(
        xmin=np.datetime64('2022-06-01'),
        ymin=np.datetime64('2022-10-01'),
        ymax=np.datetime64('2023-06-01'),
    )

plt.grid(True, color='k', linestyle=":", alpha=0.5)

# plt.twinx()

# plt.plot(dates, (t-dates).astype(float))
# plt.ylabel("Hat time minus now time (days)")

plt.legend(loc='upper left')


plt.show()
