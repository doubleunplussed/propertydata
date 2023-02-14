import json
from pathlib import Path

import numpy as np

import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

munits.registry[np.datetime64] = mdates.ConciseDateConverter()


def get_reported_and_withdrawn(city):
    data = json.loads(Path('data.json').read_text('utf8'))
    dates = []
    reported = []
    sold = []
    for date, data_by_city in data.items():
        if not data_by_city:
            continue
        reported.append(int(data_by_city[city]['reported']))
        sold.append(int(data_by_city[city]['withdrawn']))
        dates.append(np.datetime64(date))

    return np.array(dates), np.array(reported), np.array(sold)


def get_national_reported_and_withdrawn():
    data = json.loads(Path('data.json').read_text('utf8'))
    dates = []
    reported = []
    sold = []
    for date, data_by_city in data.items():
        if not data_by_city:
            continue
        total_reported = 0
        total_sold = 0
        for city_data in data_by_city.values():
            total_reported += int(city_data['reported'])
            total_sold += int(city_data['withdrawn'])

        dates.append(np.datetime64(date))
        reported.append(total_reported)
        sold.append(total_sold)

    return np.array(dates), np.array(reported), np.array(sold)


def get_four_week_withdrawn_rate(city):
    # actually just one week if WEEKLY is set
    N = 1 if WEEKLY else 4
    dates, reported, withdrawn = get_reported_and_withdrawn(city)
    four_week_reported = reported.cumsum()[N:] - reported.cumsum()[:-N]
    four_week_withdrawn = withdrawn.cumsum()[N:] - withdrawn.cumsum()[:-N]
    return dates[N:], 100 * four_week_withdrawn / four_week_reported


def get_national_four_week_withdrawn_rate():
    # actually just one week if WEEKLY is set
    N = 1 if WEEKLY else 4
    dates, reported, withdrawn = get_national_reported_and_withdrawn()
    four_week_reported = reported.cumsum()[N:] - reported.cumsum()[:-N]
    four_week_withdrawn = withdrawn.cumsum()[N:] - withdrawn.cumsum()[:-N]
    return dates[N:], 100 * four_week_withdrawn / four_week_reported


def get_reported_and_sold(city):
    data = json.loads(Path('data.json').read_text('utf8'))
    dates = []
    reported = []
    sold = []
    for date, data_by_city in data.items():
        if not data_by_city:
            continue
        reported.append(int(data_by_city[city]['reported']))
        sold.append(int(data_by_city[city]['sold']))
        dates.append(np.datetime64(date))

    return np.array(dates), np.array(reported), np.array(sold)


def get_national_reported_and_sold():
    data = json.loads(Path('data.json').read_text('utf8'))
    dates = []
    reported = []
    sold = []
    for date, data_by_city in data.items():
        if not data_by_city:
            continue
        total_reported = 0
        total_sold = 0
        for city_data in data_by_city.values():
            total_reported += int(city_data['reported'])
            total_sold += int(city_data['sold'])

        dates.append(np.datetime64(date))
        reported.append(total_reported)
        sold.append(total_sold)

    return np.array(dates), np.array(reported), np.array(sold)


def get_prices(city):
    import pandas as pd

    keys = {
        'Sydney': 'Sydney(SYDD)',
        'Melbourne': 'Melbourne (MELD)',
        'Brisbane': 'Brisbane inc Gold Coast (BRID)',
        'Adelaide': 'Adelaide (ADED)',
    }
    df = pd.read_csv('../corelogic/data.csv')
    dates = np.array([np.datetime64(d) for d in df['Date']])[::-1]
    index = np.array(df[keys[city]])[::-1]
    return dates, index


def get_national_prices():
    import pandas as pd

    df = pd.read_csv('../corelogic/data.csv')
    dates = np.array([np.datetime64(d) for d in df['Date']])[::-1]
    index = np.array(df['5 Cap City Aggregate (AUSD)'])[::-1]
    return dates, index


def padNaN(dates, rates):
    """Insert NaN values in the clearance rates dataset whenever there are gaps"""
    data = {d: r for d, r in zip(dates, rates)}

    all_dates = np.arange(dates[0], dates[-1] + 7, 7)
    all_rates = np.zeros(len(all_dates))
    for i, d in enumerate(all_dates):
        all_rates[i] = data.get(d, np.NaN)

    return all_dates, all_rates


# If WEEKY is True, then we show the weekly withdrawn rate, rather than the 4-week rate
WEEKLY = False
# WEEKLY = True

START_DATE = np.datetime64('2021-01-01')
# START_DATE = np.datetime64('2022-01-01')
SPAN_FACTOR = 1.5

# START_DATE = np.datetime64('2018-04-07')
# SPAN_FACTOR = 2.0

dates, rates = get_national_four_week_withdrawn_rate()
all_dates, all_rates = padNaN(dates, rates)
price_dates, prices = get_national_prices()
volume_dates, _, volume = get_national_reported_and_sold()

plt.plot(
    all_dates,
    all_rates,
    label="Weekly withdrawal rate" if WEEKLY else "Four-week withdrawal rate",
)
lower = np.percentile(rates[dates > START_DATE], 5)
upper = np.percentile(rates[dates > START_DATE], 95)
middle = (lower + upper) / 2
span = SPAN_FACTOR * (upper - lower)
plt.axis(
    xmin=START_DATE,
    xmax=np.datetime64('2023-06-01'),
    ymin=middle - span,
    ymax=middle + span,
)
plt.ylabel("Withdrawal rate (%)")

ax1 = plt.gca()
ax2 = plt.twinx()
price_changes = 100 * (prices[30:] / prices[:-30] - 1)
plt.plot(
    price_dates[30:],
    price_changes,
    color='C1',
    label="30-day price change",
)


lower = np.percentile(price_changes[price_dates[30:] > START_DATE], 25)
upper = np.percentile(price_changes[price_dates[30:] > START_DATE], 75)
middle = (lower + upper) / 2
span = SPAN_FACTOR * (upper - lower)

plt.bar(
    volume_dates,
    volume / volume[volume_dates > START_DATE].max(),
    width=5,
    bottom=middle - span,
    color="C2",
    label="Weekly volume",
)

plt.axis(
    ymin=middle - span,
    ymax=middle + span,
)
plt.ylabel("30-day price change (%)")
plt.title(
    "\n".join(
        [
            f"Domain {'weekly' if WEEKLY else 'four-week'} auction withdrawal rate",
            "vs CoreLogic 5-capital cities index 30-day change",
        ]
    )
)
han1, lab1 = ax1.get_legend_handles_labels()
han2, lab2 = ax2.get_legend_handles_labels()
ax2.legend(han1 + han2, lab1 + lab2, loc='upper right')

plt.tight_layout()


for city in ['Sydney', 'Melbourne', 'Adelaide', 'Brisbane']:
    plt.figure()
    dates, rates = get_four_week_withdrawn_rate(city)
    all_dates, all_rates = padNaN(dates, rates)
    price_dates, prices = get_prices(city)
    volume_dates, _, volume = get_reported_and_sold(city)
    plt.plot(
        all_dates,
        all_rates,
        label="Weekly withdrawal rate" if WEEKLY else "Four-week withdrawal rate",
    )

    lower = np.percentile(rates[dates > START_DATE], 25)
    upper = np.percentile(rates[dates > START_DATE], 75)
    middle = (lower + upper) / 2
    span = SPAN_FACTOR * (upper - lower)

    plt.axis(
        xmin=START_DATE,
        xmax=np.datetime64('2023-06-01'),
        ymin=middle - span,
        ymax=middle + span,
    )
    plt.ylabel("Withdrawal rate (%)")

    ax1 = plt.gca()
    ax2 = plt.twinx()
    plt.plot(
        price_dates[30:],
        100 * (prices[30:] / prices[:-30] - 1),
        color='C1',
        label="30-day price change",
    )

    lower = np.percentile(price_changes[price_dates[30:] > START_DATE], 25)
    upper = np.percentile(price_changes[price_dates[30:] > START_DATE], 75)
    middle = (lower + upper) / 2
    span = SPAN_FACTOR * (upper - lower)

    plt.bar(
        volume_dates,
        volume / volume[volume_dates > START_DATE].max(),
        width=5,
        bottom=middle - span,
        color="C2",
        label="Weekly volume",
    )

    plt.axis(
        ymin=middle - span,
        ymax=middle + span,
    )
    plt.ylabel("30-day price change (%)")
    plt.title(
        "\n".join(
            [
                f"{city} Domain {'weekly' if WEEKLY else 'four-week'} auction withdrawal rate",
                "vs CoreLogic index 30-day change",
            ]
        )
    )
    han1, lab1 = ax1.get_legend_handles_labels()
    han2, lab2 = ax2.get_legend_handles_labels()
    ax2.legend(han1 + han2, lab1 + lab2, loc='upper right')
    plt.tight_layout()

plt.show()
