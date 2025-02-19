import requests
import io
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat
import time
import xarray as xr


def fetch_raw(site, start_date, end_date, save_path):
    """Download just raw data"""
    url = "https://cloudnet.fmi.fi/api/raw-files"
    params = {
        "dateFrom": start_date,
        "dateTo": end_date,
        "site": site,
        "instrument": "cl61d",
    }
    metadata = requests.get(url, params).json()
    # for row in metadata:
    #     raw(row, save_path)
    #     break
    with ThreadPoolExecutor() as exe:
        exe.map(raw, metadata, repeat(save_path))


def raw(row, save_path):
    if "live" in row["filename"]:
        i = 0
        if int(row["size"]) < 100000:
            return None
        while True:
            try:
                print(row["filename"])
                bad_file = False
                res = requests.get(row["downloadUrl"])
                file_name = save_path + "/" + row["filename"]
                with open(file_name, "wb") as f:
                    f.write(res.content)
            except ValueError as error:
                i += 1
                print(i)
                if i > 50:
                    print("skip")
                    break
                print(error)
                time.sleep(1)
                continue
            except (OSError, KeyError):
                bad_file = True
                print("Bad file")
                break
            break
        if bad_file:
            return None


def response_raw(res):
    """return data in netcdf"""
    df = xr.open_groups(io.BytesIO(res.content))
    if "/diagnostics" in df.keys():
        df["/"] = df["/"].swap_dims({"profile": "time"})
    return df["/"]
