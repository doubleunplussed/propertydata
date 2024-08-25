import json
from pathlib import Path

import numpy as np

import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

munits.registry[np.datetime64] = mdates.ConciseDateConverter()


def get_prices(city, prelim=False):
    data = json.loads(Path('pricedata.json').read_text('utf8'))

    dates = []
    houses = []
    units = []

    first_date = list(sorted(data))[0]
    last_date = list(sorted(data))[-1]

    suffix = 'prelim' if prelim else 'final'
    for date, data_by_city in data.items():
        if prelim and date == first_date:
            continue
        if not prelim and date == last_date:
            continue
        dates.append(np.datetime64(date))
        houses.append(data_by_city[city][f'houses_{suffix}'])
        units.append(data_by_city[city][f'units_{suffix}'])

    return np.array(dates), np.array(houses), np.array(units)


def get_corelogic_prices(city):
    import pandas as pd

    keys = {
        'Sydney': 'Sydney(SYDD)',
        'Melbourne': 'Melbourne (MELD)',
        'Brisbane': 'Brisbane inc Gold Coast (BRID)',
        'Adelaide': 'Adelaide (ADED)',
        'Perth': 'Perth (PERD)',
        'Combined Capitals': '5 Cap City Aggregate (AUSD)',
    }
    df = pd.read_csv('../corelogic/data.csv')
    dates = np.array([np.datetime64(d) for d in df['Date']])[::-1]
    index = np.array(df[keys[city]])[::-1]
    return dates, index


# START_DATE = np.datetime64('2021-01-01')

# Whether to plot Domain data by publication date, or by contract date
DATE_SCHEME = "publication"
# DATE_SCHEME = "contract"

if DATE_SCHEME == "publication":
    DOMAIN_PRELIM_OFFSET = 30 + 25
    DOMAIN_FINAL_OFFSET = 30 + 90 + 25
else:
    DOMAIN_PRELIM_OFFSET = -15
    DOMAIN_FINAL_OFFSET = -15

for CITY in [
    "Combined Capitals",
    # "Sydney",
    # "Melbourne",
    # "Adelaide",
    # "Brisbane",
    # "Perth",
]:
    dates_prelim, houses_prelim, units_prelim = get_prices(CITY, prelim=True)
    dates_final, houses_final, units_final = get_prices(CITY, prelim=False)

    HOUSES_FRAC = 0.726 # 72.6% of dwellings are houses
    combined_prelim = HOUSES_FRAC * houses_prelim + (1 - HOUSES_FRAC) * units_prelim 
    combined_final = HOUSES_FRAC * houses_final + (1 - HOUSES_FRAC) * units_final 

    corelogic_dates, corelogic_index = get_corelogic_prices(CITY)

    corelogic_prices = (
        corelogic_index * np.nanmax(combined_final) / corelogic_index.max()
    )

    plt.plot(
        dates_prelim.astype('datetime64[D]') + DOMAIN_PRELIM_OFFSET,
        houses_prelim / 1000,
        label="Domain houses prelim",
    )

    plt.plot(
        dates_final.astype('datetime64[D]') + DOMAIN_FINAL_OFFSET,
        houses_final / 1000,
        label="Domain houses final",
    )

    # plt.plot(
    #     dates_prelim.astype('datetime64[D]') + DOMAIN_PRELIM_OFFSET,
    #     units_prelim / 1000,
    #     label="Domain units prelim",
    # )

    # plt.plot(
    #     dates_final.astype('datetime64[D]') + DOMAIN_FINAL_OFFSET,
    #     units_final / 1000,
    #     label="Domain units final",
    # )

    # plt.plot(
    #     dates_prelim.astype('datetime64[D]') + DOMAIN_PRELIM_OFFSET,
    #     combined_prelim / 1000,
    #     label="Domain prelim",
    # )

    # plt.plot(
    #     dates_final.astype('datetime64[D]') + DOMAIN_FINAL_OFFSET,
    #     combined_final / 1000,
    #     label="Domain final",
    # )

    # plt.plot(
    #     corelogic_dates,
    #     corelogic_prices / 1000,
    #     label="CoreLogic",
    # )

    plt.ylabel("Stratified median price ($k)")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    houses_prelim_change = 100 * ((houses_prelim[1:] / houses_prelim[:-1]) - 1)
    houses_final_change = 100 * ((houses_final[1:] / houses_final[:-1]) - 1)

    units_prelim_change = 100 * ((units_prelim[1:] / units_prelim[:-1]) - 1)
    units_final_change = 100 * ((units_final[1:] / units_final[:-1]) - 1)

    combined_prelim_change = 100 * ((combined_prelim[1:] / combined_prelim[:-1]) - 1)
    combined_final_change = 100 * ((combined_final[1:] / combined_final[:-1]) - 1)
      
    # plt.plot(
    #     dates_prelim[1:].astype('datetime64[D]') + DOMAIN_PRELIM_OFFSET,
    #     houses_prelim_change,
    #     label="Domain houses prelim",
    # )
    # plt.plot(
    #     dates_final[1:].astype('datetime64[D]') + DOMAIN_FINAL_OFFSET,
    #     houses_final_change,
    #     label="Domain houses final",
    # )
    # plt.plot(
    #     dates_prelim[1:].astype('datetime64[D]') + DOMAIN_PRELIM_OFFSET,
    #     units_prelim_change,
    #     label="Domain units prelim",
    # )
    # plt.plot(
    #     dates_final[1:].astype('datetime64[D]') + DOMAIN_FINAL_OFFSET,
    #     units_final_change,
    #     label="Domain units final",
    # )
    plt.plot(
        dates_prelim[1:].astype('datetime64[D]') + DOMAIN_PRELIM_OFFSET,
        combined_prelim_change,
        label="Domain prelim",
    )
    # plt.plot(
    #     dates_final[1:].astype('datetime64[D]') + DOMAIN_FINAL_OFFSET,
    #     combined_final_change,
    #     label="Domain final",
    # )

    corelogic_price_changes = 100 * (corelogic_index[90:] / corelogic_index[:-90] - 1)
    plt.plot(
        corelogic_dates[90:],
        corelogic_price_changes,
        # color='C1',
        label="CoreLogic",
    )

    plt.ylabel("QoQ change (%)")
    plt.title(f"{CITY} Domain stratified median vs CoreLogic hedonic")
    # plt.title(f"{CITY} Domain stratified median houses and units")
    plt.grid(True, color='k', linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

plt.show()
