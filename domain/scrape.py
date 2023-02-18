import json
from pathlib import Path

import requests
import numpy as np

def getcontents(html, classlabel):
    results = []
    for item in html.split(classlabel)[1:]:
        results.append(item.split('>')[1].split('<')[0])
    return results

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    }

CITY = 'class="css-1l1upik"'
SCHEDULED = 'data-label="Auctions scheduled"'
REPORTED = 'data-label="Auctions reported"'
CLEARANCE_RATE = 'data-label="Clearance rate"'
SOLD = 'data-label="Sold"'
WITHDRAWN = 'data-label="Withdrawn"'
PASSED_IN = 'data-label="Passed in"'

# The latest data is always preliminary. Thus when updating scraped data, the old URLs
# will contain updated data, so we need to go back a week or two prior to whenever we
# last scraped, in order to ensure we get the final data for all old weeks. So if you're
# running this script, you should set LAST_DATE to the most recent date Domain has data
# for, and FIRST_DATE to a couple of weeks prior to whenever the data was last scraped.


LAST_DATE = np.datetime64('2023-02-18')
# FIRST_DATE = np.datetime64("2018-04-07")
FIRST_DATE = LAST_DATE - 5 * 7

DATAFILE = Path('data.json')
HTMLDIR = Path('html')
HTMLDIR.mkdir(exist_ok=True)

try:
    data = json.loads(DATAFILE.read_text('utf8'))
except FileNotFoundError:
    data = {}

date = LAST_DATE
while date >= FIRST_DATE:
    print(date)
    url = f"https://www.domain.com.au/auction-results/national/{date}"
    html = requests.get(url, headers=headers).text
    (HTMLDIR / f"{date}.html").write_text(html, 'utf8')
    html =  (HTMLDIR / f"{date}.html").read_text('utf8')
    cities = getcontents(html, CITY)
    scheduled = getcontents(html, SCHEDULED)
    reported = getcontents(html, REPORTED)
    clearance_rate = getcontents(html, CLEARANCE_RATE)
    sold = getcontents(html, SOLD)
    withdrawn = getcontents(html, WITHDRAWN)
    passed_in = getcontents(html, PASSED_IN)

    data[str(date)] = {
        c: {
            'scheduled': sc,
            'reported': r,
            'clearance_rate': cr,
            'sold': s,
            'withdrawn': w,
            'passed_in': p,
        }
        for c, sc, r, cr, s, w, p in zip(
            cities, scheduled, reported, clearance_rate, sold, withdrawn, passed_in
        )
    }

    date -= 7

DATAFILE.write_text(json.dumps(data, indent=4, sort_keys=True), 'utf8')
