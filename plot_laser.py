import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.dates as mdates
import locale

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


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
            .between_time("22:00", "23:59")
            .resample("1h")
            .mean()
            .reset_index()
        )
        diag = pd.concat([diag, df_1h])
    diag = diag.reset_index(drop=True)
    return diag


def read_noise(site):
    noise = pd.DataFrame({})
    for file in glob.glob(f"/media/viet/CL61/{site}/Noise/*.csv"):
        df = pd.read_csv(file)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[df["datetime"] > "2000-01-01"]
        df = df.groupby("range").get_group("(6000, 8000]")
        df_1h = (
            df.set_index("datetime")
            .between_time("22:00", "23:59")
            .resample("1h")
            .mean(numeric_only=True)
            .reset_index()
        )
        noise = pd.concat([noise, df_1h])
    noise = noise.reset_index(drop=True)
    return noise


# %%
fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
for ax, site in zip(
    axes.flatten(), ["hyytiala", "kenttarova", "vehmasmaki", "lindenberg"]
):
    diag = read_diag(site)
    noise = read_noise(site)
    df = diag.merge(noise, on="datetime", how="outer")
    df = df.dropna(subset=["laser_power_percent", "co_std"])
    integration_files = glob.glob(f"/media/viet/CL61/{site}/Integration/*.csv")
    df_integration = pd.concat(
        [pd.read_csv(x) for x in integration_files], ignore_index=True
    )
    df_integration["datetime"] = pd.to_datetime(
        df_integration["datetime"], format="mixed"
    )
    df_integration["date"] = df_integration["datetime"].dt.date
    df_integration.drop(columns=["datetime"], inplace=True)
    df["date"] = df.datetime.dt.date
    df = df.merge(df_integration, on="date", how="outer")
    p = ax.scatter(
        df["laser_power_percent"],
        (df["co_std"] ** 2) * df["integration"],
        c=mdates.date2num(df.datetime),
        s=1,
    )
    cbar = fig.colorbar(p, ax=ax)
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    cbar.ax.yaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.set_ylabel(r"$\sigma²_{ppol/r^2} \times t_{integration}$")
    ax.set_xlabel("Laser power (%)")
    ax.grid()
    ax.set_title(site, weight="bold")
axes[0, 0].set_ylim([0, 6e-27])
axes[0, 1].set_ylim([0, 1e-25])
axes[1, 0].set_ylim([0, 5e-27])
fig.savefig("/media/viet/CL61/img/laser_power.png", dpi=600, bbox_inches="tight")
