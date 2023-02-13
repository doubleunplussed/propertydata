import json
from pathlib import Path
from datetime import datetime
import time

import requests
import numpy as np

def dollars(s):
    if s in ['-', 'snr']:
        return np.nan
    return int(s.strip().replace('$', '').replace(',', ''))

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }

TABLE_START = '"tableData":{'

CITIES = [
    'Sydney',
    'Melbourne',
    'Brisbane',
    'Adelaide',
    'Canberra',
    'Perth',
    'Hobart',
    'Darwin',
    'Combined Capitals',
]

LAST_DATE = np.datetime64('2022-12')
FIRST_DATE = np.datetime64("2017-03")

DATAFILE = Path('pricedata.json')
HTMLDIR = Path('pricehtml')
HTMLDIR.mkdir(exist_ok=True)

try:
    data = json.loads(DATAFILE.read_text('utf8'))
except FileNotFoundError:
    data = {}

date = LAST_DATE
while date >= FIRST_DATE:
    print(date)
    # datestr = date.astype(datetime).strftime("%B-%Y").lower()
    # url = f"https://www.domain.com.au/research/house-price-report/{datestr}/"
    # response = requests.get(url, headers=headers)
    # (HTMLDIR / f"{date}.html").write_text(response.text, 'utf8')

    html =  (HTMLDIR / f"{date}.html").read_text('utf8')

    tablejson = html.split(TABLE_START)[1].split('}')[0]
    table = json.loads(f"{{{tablejson}}}")

    data.setdefault(str(date), {})
    data.setdefault(str(date - 3), {})

    for city in CITIES:
        key = city
        if city == 'Combined Capitals':
            for altkey in ['Combined capitals', 'National']:
                if altkey in tablejson:
                    key = altkey
                    break

        data[str(date)].setdefault(city, {})
        data[str(date - 3)].setdefault(city, {})

        hprelim, hfinal = [x[1:3] for x in table['Houses'] if x[0] == key][0]
        data[str(date)][city]['houses_prelim'] = dollars(hprelim)
        data[str(date - 3)][city]['houses_final'] = dollars(hfinal)

        try:
            uprelim, ufinal = [x[1:3] for x in table['Units'] if x[0] == key][0]
        except IndexError:
            data[str(date)][city]['units_prelim'] = np.nan
            data[str(date - 3)][city]['units_final'] = np.nan
        else:
            data[str(date)][city]['units_prelim'] = dollars(uprelim)
            data[str(date - 3)][city]['units_final'] = dollars(ufinal)

    date -= 3

DATAFILE.write_text(json.dumps(data, indent=4, sort_keys=True), 'utf8')
