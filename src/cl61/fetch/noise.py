import requests
import pandas as pd
import time
from cl61.fetch.response import response
from cl61.func.noise import noise_detection


def fetch_noise(site, start_date, end_date, save_path):
    url = "https://cloudnet.fmi.fi/api/raw-files"
    pr = pd.period_range(start=start_date, end=end_date, freq="D")

    for i in pr:
        idate = i.strftime("%Y-%m-%d")
        print(idate)
        result = pd.DataFrame({})
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
                        df, _ = response(res)
                        df = noise_detection(df)
                        df_noise = df.where(df["noise"])
                        df_noise["p_pol"] = df_noise["p_pol"] / (df["range"] ** 2)
                        df_noise["x_pol"] = df_noise["x_pol"] / (df["range"] ** 2)
                        grp_range = df_noise[["p_pol", "x_pol"]].groupby_bins(
                            "range", [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000]
                        )

                        grp_mean = grp_range.mean(dim=["range", "time"])
                        grp_std = grp_range.std(dim=["range", "time"])

                        result_ = pd.DataFrame(
                            {
                                "datetime": df.time[0].values,
                                "co_mean": grp_mean["p_pol"],
                                "co_std": grp_std["p_pol"],
                                "cross_mean": grp_mean["x_pol"],
                                "cross_std": grp_std["x_pol"],
                                "range": grp_mean.range_bins.values.astype(str),
                            }
                        )

                        result = pd.concat([result_, result])
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
        result.to_csv(save_path + i.strftime("%Y%m%d") + "_" + "noise.csv", index=False)
