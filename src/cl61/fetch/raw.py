import requests
import os


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
    for row in metadata:
        file_name = row["filename"]
        file_path = save_path + "/" + row["filename"]
        if os.path.isfile(file_path):
            print("Done")
            continue
        if "live" in file_name:
            print(file_name)
            res = requests.get(row["downloadUrl"])
            with open(file_path, "wb") as f:
                f.write(res.content)
