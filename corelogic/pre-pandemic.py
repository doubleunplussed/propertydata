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
        # '5 capital city aggregate': np.array(df['5 Cap City Aggregate (AUSD)'])[::-1],
        'Sydney': np.array(df['Sydney(SYDD)'])[::-1],
        'Melbourne': np.array(df['Melbourne (MELD)'])[::-1],
        'Brisbane': np.array(df['Brisbane inc Gold Coast (BRID)'])[::-1],
        'Adelaide': np.array(df['Adelaide (ADED)'])[::-1],
        'Perth': np.array(df['Perth (PERD)'])[::-1],
        '5-city aggregate': np.array(df['5 Cap City Aggregate (AUSD)'])[::-1],
    }

N = 30

def n_day_change(x, prepend=None):
    if prepend is not None:
        x = np.concatenate([prepend, x])
    return 100 * (x[N:] / x[:-N] - 1)


all_dates, data = get_index_data()

# all_dates = all_dates[:-30]
# for CITY in data:
#     data[CITY] = data[CITY][:-30]

for FORTY_PERCENT in [True, False]:
    plt.figure()
    for CITY in data:
        # CITY = 'Sydney'

        index = data[CITY]

        peak_date = all_dates[index.argmax()]
        if FORTY_PERCENT:
            # Forty percent of max
            TARGET = 0.6 * index.max()
        else:
            # 2020 max:
            TARGET = index[all_dates <= np.datetime64('2020-12-31')].max()
            # # EOY 2020:
            # TARGET = index[all_dates.searchsorted(np.datetime64('2020-12-31'))]
        
        peak_index = index[all_dates.searchsorted(peak_date)]

        velocity = n_day_change(index)[all_dates[N:] > peak_date + N]
        index = index[all_dates > peak_date + N]
        dates = all_dates[all_dates > peak_date + N]

        # plt.plot(dates, velocity)
        # plt.show()



        k = (1 + velocity / 100) ** (1 / N) - 1
        t = dates + (1 / k * np.log(TARGET / index)).astype(int)
        # t = 1 / k * np.log(EOY_2020_index / index)

        plt.plot(dates, t, label=f"{CITY} ({t[-1]})")

    if FORTY_PERCENT:
        plt.title("Date of 40% loss @ current MoM decline")
    else:
        plt.title("Date pandemic-gains erased @ current MoM decline")

    plt.ylabel("Projected date")
    plt.axis(
        xmin=np.datetime64('2022-06-01'),
        ymin=np.datetime64('2023-01-01'),
        ymax=np.datetime64('2026-06-01') if FORTY_PERCENT else np.datetime64('2025-01-01'),
    )
    plt.grid(True, color='k', linestyle=":", alpha=0.5)
    plt.legend()


plt.show()
