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
        'Sydney': np.array(df['Sydney(SYDD)'])[::-1],
        'Melbourne': np.array(df['Melbourne (MELD)'])[::-1],
        'Brisbane': np.array(df['Brisbane inc Gold Coast (BRID)'])[::-1],
        'Adelaide': np.array(df['Adelaide (ADED)'])[::-1],
        'Perth': np.array(df['Perth (PERD)'])[::-1],
        '5-city aggregate': np.array(df['5 Cap City Aggregate (AUSD)'])[::-1],
    }


dates, data = get_index_data()

N = 30
M = 28


def m_day_average(x):
    ret = np.cumsum(x, dtype=float)
    ret[M:] = ret[M:] - ret[:-M]
    return ret / M


def n_day_change(x, prepend=None):
    if prepend is not None:
        x = np.concatenate([prepend, x])
    return 100 * (x[N:] / x[:-N] - 1)



for SMOOTHED in [True, False]:
    plt.figure(figsize=(7, 6))

    for city, index in data.items():

        change = 100 * (index[-1] / index.max() - 1)

        if SMOOTHED:
            velocity = n_day_change(m_day_average(index))
            accel = velocity[N:] - velocity[:-N]
            plt.plot(
                dates[N:],
                velocity,
                linewidth=3,
                # color='k',
                label=f'{city} ({change:+.1f}%, {velocity[-1]:+.2f} %/m, {accel[-1]:+.2f} %/m/m)',
                alpha=0.7,
            )
        else:
            velocity = n_day_change(index)
            accel = velocity[N:] - velocity[:-N]
            plt.plot(
                dates[N:],
                velocity,
                linewidth=3,
                # color='k',
                # label=f'{city} ({change:+.1f}%, {velocity[-1]:+.2f} %/m, {accel[-1]:+.2f} %/m/m)',
                label=f'{city} ({change:+.1f}%, {velocity[-1]:+.2f} %/m)',
                alpha=0.7,
            )


    plt.grid(True, color='k', linestyle="-", alpha=0.25)
    plt.ylabel(f'{N}-day change (%)' + (' (smoothed)' if SMOOTHED else ''))
    plt.legend(loc='upper right', prop={'size': 9})
    plt.axis(
        # xmin=dates.min() + 30,
        xmin=np.datetime64('2022-01-01'),
        xmax=np.datetime64('2023-06-01'),
        ymin=-2.5,
        ymax=3.0,
    )
    plt.axhline(0, color='k')
    plt.title(f'CoreLogic indices {N}-day change')


    plt.figure(figsize=(7, 6))

    for city, index in data.items():
        if SMOOTHED:
            velocity = n_day_change(m_day_average(index))
            accel = velocity[N:] - velocity[:-N]
            plt.plot(
                dates[2 * N :],
                accel,
                linewidth=3,
                # color='k',
                label=f'{city} ({velocity[-1]:+.2f} %/m, {accel[-1]:+.2f} %/m/m)',
                alpha=0.7,
            )
        else:
            velocity = n_day_change(index)
            accel = velocity[N:] - velocity[:-N]
            plt.plot(
                dates[2 * N :],
                accel,
                linewidth=3,
                # color='k',
                label=f'{city} ({velocity[-1]:+.2f} %/m, {accel[-1]:+.2f} %/m/m)',
                alpha=0.7,
            )


    plt.grid(True, color='k', linestyle="-", alpha=0.25)
    plt.ylabel(
        f'{N}-day change in {N}-day change (%)' + (' (smoothed)' if SMOOTHED else '')
    )
    plt.legend(loc='upper right', prop={'size': 9})
    plt.axis(
        # xmin=dates.min() + 30,
        xmin=np.datetime64('2022-01-01'),
        xmax=np.datetime64('2023-06-01'),
        ymin=-2.0,
        ymax=2.0,
    )
    plt.axhline(0, color='k')
    plt.title('Acceleration')



plt.figure(figsize=(7, 6))

# 2020 peak
START_IX = dates.searchsorted(np.datetime64('2020-04-22'))
# START_IX = dates.searchsorted(np.datetime64('2020-01-01'))
START_IX = 0
START_IX = dates.searchsorted(np.datetime64('2018-01-01'))

for city, index in data.items():
    referenced = 100 * index[START_IX:] / index[START_IX]
    plt.plot(
        dates[START_IX:],
        # index,
        referenced,
        linewidth=3,
        # color='k',
        label=f'{city} ({referenced[-1] - 100:+.1f}%)',
        alpha=0.7,
    )

# plt.plot([np.datetime64('2023-02-11'), np.datetime64('2023-03-11')], [185.92] * 2, 'ko')

plt.axhline(100, color='k')
plt.title(f'CoreLogic indices (indexed to 100 on {dates[START_IX]})')
# plt.title(f'CoreLogic indices (indexed to 100 at series start)')
plt.grid(True, color='k', linestyle="-", alpha=0.25)
plt.legend(loc='upper left', prop={'size': 9})
plt.axis(
    # xmin=dates.min() + 30,
    xmin=dates[START_IX] - 30,
    xmax=np.datetime64('2023-06-01'),
    # ymin=-2.0,
    # ymax=2.0,
)



plt.show()


