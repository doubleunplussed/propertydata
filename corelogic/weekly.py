from datetime import datetime, timedelta
from pathlib import Path

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


dates, data = get_index_data()

N = 7
for city in data:
    plt.figure()
    index = data[city]

    changes = []
    week_ending = dates[1::N][1:]
    week_ending = dates[::-1][::N][::-1][1:]
    for end_date in week_ending:
        start_date = end_date - N
        start_index = index[dates == start_date][0]
        end_index = index[dates == end_date][0]

        percent_change = 100 * (end_index / start_index - 1)

        changes.append(percent_change)
    plt.bar(week_ending, changes, width=N)
    # plt.bar(week_ending[6::7], changes[6::7], width=N, color='r')
    plt.grid(True, color='k', linestyle="-", alpha=0.25)
    plt.axis(xmin=np.datetime64('2022-01-01'))
    plt.ylabel('weekly change (%)')
    plt.title(city)

plt.show()
