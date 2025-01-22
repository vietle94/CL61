import xarray as xr
import io
import pandas as pd


def response(res):
    # return data in netcdf and diagnostics in csv
    df = xr.open_groups(io.BytesIO(res.content))
    if "/diagnostics" in df.keys():
        df["/"] = df["/"].swap_dims({"profile": "time"})
        diag = pd.DataFrame([df["/diagnostics"].attrs])
        diag["datetime"] = df["/"].time[0].values

    elif "Timestamp" in df["/monitoring"].attrs:
        monitoring = pd.DataFrame([df["/monitoring"].attrs])
        monitoring = monitoring.rename({"Timestamp": "datetime"}, axis=1)
        monitoring.datetime = monitoring.datetime.astype(float)
        monitoring["datetime"] = pd.to_datetime(monitoring["datetime"], unit="s")

        status = pd.DataFrame([df["/status"].attrs])
        status = status.rename({"Timestamp": "datetime"}, axis=1)
        status.datetime = status.datetime.astype(float)
        status["datetime"] = pd.to_datetime(status["datetime"], unit="s")
        diag = monitoring.merge(status)
    else:
        monitoring = df["/monitoring"].to_dataframe().reset_index()
        monitoring = monitoring.rename({"time": "datetime"}, axis=1)

        status = df["/status"].to_dataframe().reset_index()
        status = status.rename({"time": "datetime"}, axis=1)
        diag = monitoring.merge(status)

    return df["/"], diag
