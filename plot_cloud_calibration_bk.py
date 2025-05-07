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
    break

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
fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
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

# %%
def read_diag(site):
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
            .resample("5min")
            .mean()
            .reset_index()
        )
        diag = pd.concat([diag, df_1h])
    diag = diag.reset_index(drop=True)
    return diag

# %%
cmap = plt.get_cmap("viridis")
cmap.set_bad("grey")
fig, axes = plt.subplots(4, 1, figsize=(9, 6), constrained_layout=True, sharey=True, sharex=True)
for site, ax, lim in zip(
    ["vehmasmaki", "hyytiala", "kenttarova", "lindenberg"],
    axes.flatten(),
    [12, 10, 10, 3],
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

    df_full = cloud.merge(diag, how="left")

    p = ax.scatter(
        df_full.datetime,
        df_full["c"],
        c=df_full["laser_power_percent"],
        plotnonfinite=True,
        vmin=0,
        vmax=100,
        s=1,
        cmap=cmap,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator([6, 12]))
    ax.grid()
    ax.set_ylabel("c")
    ax.axhline(1, color="grey", linestyle="--")
ax.set_xlim(right=pd.to_datetime("2024-12-30"))
fig.colorbar(p, ax=axes.ravel().tolist(), label="Laser power (%)")
ax_flat = axes.flatten()
ax_flat[0].axvspan("2022-06-15", "2023-04-27", color="C0", alpha=0.2, label="1.1.10")
ax_flat[0].axvspan("2023-04-28", "2025-01-01", color="C1", alpha=0.2, label="1.2.7")
ax_flat[1].axvspan("2022-11-21", "2023-11-22", color="C0", alpha=0.2, label="1.1.10")
ax_flat[1].axvspan("2023-11-23", "2025-01-01", color="C1", alpha=0.2, label="1.1.10")
ax_flat[2].axvspan("2023-06-21", "2025-01-01", color="C1", alpha=0.2, label="1.1.7")
ax_flat[3].axvspan("2024-03-01", "2025-01-01", color="C0", alpha=0.2, label="1.1.10")

ax.set_ylim(0, 10)
for n, ax_ in enumerate(axes.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
    ax_.legend(loc="upper left")
    

fig.savefig(
    "/media/viet/CL61/img/calibration_factor_ts2.png", bbox_inches="tight", dpi=600
)

