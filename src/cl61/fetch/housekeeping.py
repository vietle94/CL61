import requests
import pandas as pd
import time
from cl61.fetch.response import response


def fetch_housekeeping(site, start_date, end_date, save_path):
    url = "https://cloudnet.fmi.fi/api/raw-files"
    pr = pd.period_range(start=start_date, end=end_date, freq="D")

    for i in pr:
        idate = i.strftime("%Y-%m-%d")
        print(idate)
        diag = pd.DataFrame({})
        params = {
            "dateFrom": idate,
            "dateTo": idate,
            "site": site,
            "instrument": "cl61d",
        }
        metadata = requests.get(url, params).json()
        if not metadata:
            continue

        for row in metadata:
            if "live" in row["filename"]:
                if int(row["size"]) < 100000:
                    continue
                while True:
                    try:
                        print(row["filename"])
                        bad_file = False
                        res = requests.get(row["downloadUrl"])
                        _, diag_ = response(res)
                        diag = pd.concat([diag_, diag])
                    except ValueError as error:
                        print(error)
                        time.sleep(1)
                        continue
                    except OSError:
                        bad_file = True
                        print("Bad file")
                        break
                    break
                if bad_file:
                    continue

        print("saving")
        diag.to_csv(save_path + i.strftime("%Y%m%d") + "_" + "diag.csv", index=False)
