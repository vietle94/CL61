import pandas as pd
import requests
from cl61.fetch.utils import process_metadata, response, process_metadata_child
from cl61.func.calibration_cloud import calibration_etaS
import xarray as xr
import io
import numpy as np

ref = np.load("cal_ref.npy")


def fetch_processing(func, site, start_date, end_date, save_path):
    url = "https://cloudnet.fmi.fi/api/raw-files"
    pr = pd.period_range(start=start_date, end=end_date, freq="D")

    for i in pr:
        idate = i.strftime("%Y-%m-%d")
        print(idate)
        params = {
            "dateFrom": idate,
            "dateTo": idate,
            "site": site,
            "instrument": "cl61d",
        }
        metadata = requests.get(url, params).json()
        if not metadata:
            continue
        result = process_metadata(metadata, func)
        print("saving")
        result.to_csv(save_path + i.strftime("%Y%m%d") + ".csv", index=False)


def noise(res):
    from cl61.func.noise import noise_detection

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
    return result_


def housekeeping(res):
    _, diag_ = response(res)
    return diag_


def fetch_attrs(func, site, start_date, end_date, save_path):
    url = "https://cloudnet.fmi.fi/api/raw-files"
    pr = pd.period_range(start=start_date, end=end_date, freq="D")
    for i in pr:
        idate = i.strftime("%Y-%m-%d")
        print(idate)
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
                result = process_metadata_child(row, func)
                if result is not False:
                    print("saving")
                    result.to_csv(
                        save_path + i.strftime("%Y%m%d") + ".csv", index=False
                    )
                break


def integration(res):
    df = xr.open_dataset(io.BytesIO(res.content))
    integration_name = [
        "time between consecutive profiles in seconds",
        "profile_interval_in_seconds",
    ]
    result = pd.DataFrame(
        {
            "datetime": [df.time[0].values],
            "integration_t": [
                np.median(
                    np.diff(df["time"].values).astype("timedelta64[ns]")
                    / np.timedelta64(1, "s")
                )
            ],
        }
    )
    for attr in integration_name:
        if attr in df.attrs:
            result["integration"] = df.attrs[attr]
    return result


def sw_version(res):
    df = xr.open_dataset(io.BytesIO(res.content))
    if "sw_version" in df.attrs:
        return pd.DataFrame(
            {"datetime": [df.time[0].values], "sw_version": [df.attrs["sw_version"]]}
        )
    return False


def cloud_calibration(res):
    df, _ = response(res)
    df = df.sel(range=slice(1000, 4000))
    test = df.mean(dim="time")
    res = np.convolve(test.p_pol, ref, mode="same")
    return pd.DataFrame(
        {
            "datetime": [df.time[0].values],
            "cross_correlation": [res.max()],
            "etaS": [calibration_etaS(test).values],
        }
    )
