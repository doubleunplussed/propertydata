from datetime import datetime, timedelta
from pathlib import Path
import subprocess

import numpy as np


# Script to download the latest 1-year daily backseries from CoreLogic, convert to csv,
# and merge all the csvs together.
#
# requires `curl` and `xlsx2csv` command line utilities
#
# Thanks to /u/shrugmeh for the data back to 2106 - this is in "csv/long-converted.csv"


def download_data(date):
    Path('xlsx').mkdir(exist_ok=True)
    Path('csv').mkdir(exist_ok=True)
    basename = f'CoreLogic_HVI_365_days_{date:%Y%m%d}'
    xlsx = Path(f"xlsx/{basename}.xlsx")
    csv = Path(f"csv/{basename}.csv")
    url = f"https://download.rpdata.com/asx/{basename}.xlsx"
    print(url)
    if not csv.exists():
        try:
            subprocess.check_call(['curl', '--fail', url, '-o', xlsx])
        except subprocess.CalledProcessError:
            return None
        subprocess.check_call(['xlsx2csv', '--dateformat', '%Y-%m-%d', xlsx, csv])
    return csv


def download_latest():
    date = datetime.now()
    csv = download_data(date)
    if csv is None:
        # Not updated yet today - try yesterday
        date -= timedelta(days=1)
        csv = download_data(date)
    assert csv is not None
    return np.datetime64(date, 'D'), csv


def merge_csvs():
    lines = set(
        [
            "Date,Sydney(SYDD),Melbourne (MELD),Brisbane inc Gold Coast (BRID),"
            + "Adelaide (ADED),Perth (PERD),5 Cap City Aggregate (AUSD)"
        ]
    )
    for path in Path('csv').iterdir():
        lines.update(
            [
                l.strip()
                for l in path.read_text().splitlines()
                if l.strip() and l[0].isdigit()
            ]
        )
    Path('data.csv').write_text('\n'.join(reversed(sorted(lines))), encoding='utf8')


if __name__ == '__main__':
    download_latest()
    merge_csvs()
