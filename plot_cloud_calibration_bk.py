import matplotlib.pyplot as plt
import glob
from cl61.func.calibration_cloud import cloud_calibration
import matplotlib.dates as mdates
import string
import pandas as pd

# %%
fig, axes = plt.subplots(4, 1, figsize=(9, 6), constrained_layout=True, sharex=True)
for site, ax, lim in zip(
    ["vehmasmaki", "hyytiala", "kenttarova", "lindenberg"],
    axes.flatten(),
    [12, 10, 10, 3],
):
    cloud = cloud_calibration(site)
    ax.scatter(cloud.datetime, cloud["c"], alpha=0.05, s=1)
    ax.set_ylim(0, lim)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator([6, 12]))
    ax.grid()
    ax.set_ylabel("c")

for n, ax_ in enumerate(axes.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
fig.savefig(
    "/media/viet/CL61/img/calibration_factor_ts.png", bbox_inches="tight", dpi=600
)
# %%
fig, axes = plt.subplots(4, 1, figsize=(9, 6), constrained_layout=True, sharex=True)
for site, ax, lim in zip(
    ["vehmasmaki", "hyytiala", "kenttarova", "lindenberg"],
    axes.flatten(),
    [12, 10, 10, 2],
):
    cloud = cloud_calibration(site)
    file_dir = f"/media/viet/CL61/{site}/Noise/*.csv"
    files = glob.glob(file_dir)
    df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["datetime"] > "2000-01-01"]
    df = df.set_index("datetime").between_time("21:00", "23:00").reset_index()
    df = df.groupby("range").get_group("(6000, 8000]")

    cloud = (
        cloud.set_index("datetime")
        .resample("5min")
        .mean(numeric_only=True)
        .dropna()
        .reset_index()
    )
    df = (
        df.set_index("datetime")
        .resample("5min")
        .mean(numeric_only=True)
        .dropna()
        .reset_index()
    )

    df_full = cloud.merge(df)
    ax.scatter(df_full["co_std"], df_full["c"], alpha=0.5, s=1)

    ax.set_ylim(0, lim)
    ax.grid()
    ax.set_ylabel("c")

ax.set_xlabel("co_std")
ax.set_xscale("log")

for n, ax_ in enumerate(axes.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
fig.savefig(
    "/media/viet/CL61/img/calibration_factor_bk.png", bbox_inches="tight", dpi=600
)


# %%
def read_diag(site, time=("22:00", "23:59")):
    diag = pd.DataFrame({})
    for file in glob.glob(f"/media/viet/CL61/{site}/Diag/*.csv"):
        df = pd.read_csv(file)
        try:
            df = df[["datetime", "laser_power_percent", "internal_temperature"]]
        except KeyError:
            continue
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[df["datetime"] > "2000-01-01"]
        df_1h = (
            df.set_index("datetime")
            .between_time(*time)
            .resample("5min")
            .mean()
            .reset_index()
        )
        diag = pd.concat([diag, df_1h])
    diag = diag.reset_index(drop=True)
    return diag


# %%
fig, axes = plt.subplots(4, 1, figsize=(9, 6), constrained_layout=True)
for site, ax, lim in zip(
    ["vehmasmaki", "hyytiala", "kenttarova", "lindenberg"],
    axes.flatten(),
    [2, 2, 2, 2],
):
    cloud = cloud_calibration(site)
    diag = read_diag(site)

    cloud = (
        cloud.set_index("datetime")
        .resample("5min")
        .mean(numeric_only=True)
        .dropna()
        .reset_index()
    )

    df_full = cloud.merge(diag)
    ax.scatter(df_full["laser_power_percent"], df_full["c"], alpha=0.5, s=1)

    ax.set_ylim(0, lim)
    ax.grid()
    ax.set_ylabel("c")

ax.set_xlabel("laser_power_percent")

for n, ax_ in enumerate(axes.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
fig.savefig(
    "/media/viet/CL61/img/calibration_factor_laser.png", bbox_inches="tight", dpi=600
)
