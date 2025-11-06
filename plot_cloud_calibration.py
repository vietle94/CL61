import matplotlib.pyplot as plt
import glob
import matplotlib.dates as mdates
import string
import pandas as pd


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
        df_1h = df.set_index("datetime").resample("5min").mean().reset_index()
        diag = pd.concat([diag, df_1h])
    diag = diag.reset_index(drop=True)
    return diag


# %%
fig, axes = plt.subplots(4, 1, figsize=(9, 6), constrained_layout=True, sharex=True)
for site, ax, lim in zip(
    ["vehmasmaki", "hyytiala", "kenttarova", "lindenberg"],
    axes.flatten(),
    [12, 10, 10, 2],
):
    cloud = cloud_calibration(site)
    cloud = cloud[(cloud["cloud_base"] > 1000) & (cloud["cloud_base"] < 1200)]
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
fig, ax = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
for site, ax_ in zip(
    ["vehmasmaki", "hyytiala", "kenttarova", "lindenberg"],
    ax.flatten(),
):
    cloud = pd.read_csv(
        f"/media/viet/CL61/cloud_calibration/cloud_calibration_{site}.csv"
    )
    cloud["datetime"] = pd.to_datetime(cloud["datetime"])
    diag = read_diag(site)
    diag = diag[diag["datetime"] < pd.to_datetime("2025-01-01")]
    diag["month_year"] = diag["datetime"].dt.to_period("M")
    diag = diag.groupby(diag.month_year)["laser_power_percent"].mean()
    diag = diag.reset_index()
    ax_.plot(
        cloud["datetime"],
        cloud["calibration_factor"],
        ".-",
        color="C0",
    )
    ax_.set_ylabel("c", color="C0")
    ax_.tick_params(axis="y", labelcolor="C0")
    ax_.grid()
    ax_2 = ax_.twinx()
    ax_2.plot(
        diag["month_year"].dt.to_timestamp(),
        diag["laser_power_percent"],
        ".-",
        color="C1",
    )
    ax_2.set_ylabel("Laser power (%)", color="C1")
    ax_2.tick_params(axis="y", labelcolor="C1")
    ax_2.set_ylim(0, 105)
    ax_.axhline(1, color="grey", linestyle="--")
ax_flat = ax.flatten()
ax_flat[0].axvspan(
    "2022-06-15", "2023-04-27", color="tab:brown", alpha=0.2, label="1.1.10"
)
ax_flat[0].axvspan(
    "2023-04-28", "2025-01-01", color="tab:purple", alpha=0.2, label="1.2.7"
)
ax_flat[1].axvspan(
    "2022-11-21", "2023-11-22", color="tab:brown", alpha=0.2, label="1.1.10"
)
ax_flat[1].axvspan(
    "2023-11-23", "2025-01-01", color="tab:purple", alpha=0.2, label="1.2.7"
)
ax_flat[2].axvspan(
    "2023-06-21", "2025-01-01", color="tab:purple", alpha=0.2, label="1.2.7"
)
ax_flat[3].axvspan(
    "2023-07-01", "2025-01-01", color="tab:brown", alpha=0.2, label="1.1.10"
)
ax_.set_xlim(right=pd.to_datetime("2024-12-31"))
ax_.set_ylim(0.5, 2)
for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
    # ax_.legend(loc="upper left")
    ax_.tick_params(axis="x", labelrotation=45)
    ax_.yaxis.set_tick_params(labelleft=True)

handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, title="Firmware version")
fig.subplots_adjust(bottom=0.2, hspace=0.3, wspace=0.42)

fig.savefig(
    "/media/viet/CL61/img/calibration_factor_ts.png", bbox_inches="tight", dpi=600
)
# %%
