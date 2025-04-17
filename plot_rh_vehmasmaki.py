import pandas as pd
import glob
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm

# %%
file_dir = "/media/viet/CL61/studycase/vehmasmaki/20230519/"
df = xr.open_mfdataset(glob.glob(file_dir + "*.nc"))

fig, ax = plt.subplots(4, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
p = ax[0].pcolormesh(
    df["time"],
    df["range"],
    df["p_pol"].T,
    norm=LogNorm(vmin=1e-7, vmax=1e-5),
)
fig.colorbar(p, ax=ax[0], label=r"$\beta$ [m-1 sr-1]")

file_dir = "/media/viet/CL61/vehmasmaki/Diag/"
df = pd.read_csv(file_dir + "20230519.csv")
df["datetime"] = pd.to_datetime(df["datetime"])


def get_dew_point_c(t_air_c, rel_humidity):
    """Compute the dew point in degrees Celsius
    :param t_air_c: current ambient temperature in degrees Celsius
    :type t_air_c: float
    :param rel_humidity: relative humidity in %
    :type rel_humidity: float
    :return: the dew point in degrees Celsius
    :rtype: float
    """
    A = 17.27
    B = 237.7
    alpha = ((A * t_air_c) / (B + t_air_c)) + np.log(rel_humidity / 100.0)
    return (B * alpha) / (A - alpha)


dp = get_dew_point_c(df["internal_temperature"].values, df["internal_humidity"].values)

t = pd.read_csv(
    "/home/viet/Downloads/Kuopio Savilahti_ 19.5.2023 - 19.5.2023_8ae9fbe7-08db-40e7-b68f-a6cd87c8bcf1.csv"
)
t["Year"] = t["Year"].astype(str)
t["Month"] = t["Month"].astype(str).str.zfill(2)
t["Day"] = t["Day"].astype(str).str.zfill(2)
t["Time [UTC]"] = t["Time [UTC]"].astype(str)
t["datetime"] = pd.to_datetime(
    t["Year"] + "-" + t["Month"] + "-" + t["Day"] + " " + t["Time [UTC]"]
)

ax[1].plot(
    df["datetime"],
    df["window_condition"],
    ".",
)
ax[3].plot(
    df["datetime"],
    df["window_blower_heater"],
    ".",
)
ax[3].grid()
ax[2].plot(df["datetime"], dp, ".", label="Internal dew point")
ax[2].plot(t["datetime"], t["Air temperature [°C]"], ".", label="Air temperature")
ax[2].legend()
ax[2].set_ylabel("Temperature [°C]")
ax[1].set_ylabel("Window condition")
ax[3].set_ylabel("Window blower heater")
ax[1].grid()
ax[2].grid()
ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

fig.savefig(
    "/media/viet/CL61/img/window_condition_study_case_vehmasmaki.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
df = pd.DataFrame({})
for file in glob.glob("/media/viet/CL61/vehmasmaki/Diag/*.csv"):
    df_ = pd.read_csv(file)
    if "window_condition" not in df_.columns:
        continue
    df_["datetime"] = pd.to_datetime(df_["datetime"])
    df_ = df_[df_["datetime"] > "2000-01-01"]
    df_1h = (
        df_.set_index("datetime")["window_condition"]
        .resample("1h")
        .mean(numeric_only=True)
        .reset_index()
    )
    df = pd.concat([df, df_1h], ignore_index=True)

fig, ax = plt.subplots(figsize=(9, 3))
ax.scatter(df["datetime"], df["window_condition"], alpha=0.5, s=5, edgecolors="None")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.set_ylim(70, 105)
ax.set_axisbelow(True)
ax.grid()
ax.set_ylabel("Window condition")
fig.savefig(
    "/media/viet/CL61/img/window_condition_vehmasmaki.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
site = "hyytiala"
df = pd.DataFrame({})
for file in glob.glob(f"/media/viet/CL61/{site}/Diag/*.csv"):
    df_ = pd.read_csv(file)
    if "window_condition" not in df_.columns:
        continue
    df_["datetime"] = pd.to_datetime(df_["datetime"])
    df_ = df_[df_["datetime"] > "2000-01-01"]
    df_1h = (
        df_.set_index("datetime")["window_condition"]
        .resample("1h")
        .mean(numeric_only=True)
        .reset_index()
    )
    df = pd.concat([df, df_1h], ignore_index=True)

fig, ax = plt.subplots(figsize=(9, 3))
ax.scatter(df["datetime"], df["window_condition"], alpha=0.5, s=5, edgecolors="None")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.set_ylim(70, 105)
ax.set_axisbelow(True)
ax.grid()
ax.set_ylabel("Window condition")
fig.savefig(
    f"/media/viet/CL61/img/window_condition_{site}.png",
    dpi=300,
    bbox_inches="tight",
)
