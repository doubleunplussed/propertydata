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


months = [
    'Jan\n2022',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec',
    'Jan\n2023',
    'Feb',
]


dates, data = get_index_data()


def add_n_months(d, n):
    y, m, d = [int(s) for s in str(d).split('-')]
    m += n
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return np.datetime64(f'{y}-{m:02d}-{d:02d}')

PARTIAL_MONTH_METHOD = 'overlap'
# PARTIAL_MONTH_METHOD = 'extrapolate'

for city in ['Sydney', '5-city aggregate', 'Melbourne', 'Brisbane', 'Adelaide', 'Perth']:
    plt.figure(figsize=(8,6))
    index = data[city]

    changes = []
    for i in range(len(months)):
        print(f'{i=}')
        y = 2022 + i // 12
        m = i % 12 + 1

        start_of_month = np.datetime64(f'{y}-{m:02d}-01') 

        print("Month is:", months[i])

        start_date = start_of_month - 1
        end_date = add_n_months(start_of_month, 1) - 1

        extrapolation_factor = 1.0
        if i == len(months) - 1 and end_date != dates[-1]:
            # Incomplete month. How many days are in the current month?
            n = (end_date - start_date).astype(int)
            if PARTIAL_MONTH_METHOD == 'overlap':
                # Show the latest n-day change instead, where n is the number of days in
                # the current month.
                end_date = dates[-1]
                start_date = end_date - n
            elif PARTIAL_MONTH_METHOD == 'extrapolate':
                # Extrapolate the change so far this month to the whole month:
                end_date = dates[-1]
                extrapolation_factor = n / int(str(dates[-1]).split('-')[-1])

        print(f"{start_date=}")
        print(f"{end_date=}")

        start_index = index[dates == start_date][0]
        end_index = index[dates == end_date][0]

        percent_change = 100 * (end_index / start_index - 1)

        percent_change *= extrapolation_factor

        changes.append(percent_change)
        print(f"{city} 2022-{m:02d}: {percent_change:+.2f}%")

    bars = plt.bar(np.arange(len(months)), changes, tick_label=months)
    plt.bar_label(bars, fmt="%+.1f%%", padding=2, fontsize=9)
    # plt.grid(True, color='k', linestyle="-", alpha=0.25)
    plt.axhline(0, color='k')
    plt.ylabel('monthly change (%)')
    plt.title(f"CoreLogic {city} index monthly change")

plt.show()


