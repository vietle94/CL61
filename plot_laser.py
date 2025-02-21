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
            df = df[["datetime", "laser_power_percent"]]
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
        df = df.sort_values("datetime")
        integration_time = df.datetime.diff().dt.total_seconds().median()
        df["integration_time"] = integration_time
        df["co_std"] = df["co_std"] * np.sqrt(integration_time)
        df["cross_std"] = df["cross_std"] * np.sqrt(integration_time)
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
fig, axes = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
for ax, site in zip(
    axes.flatten(), ["hyytiala", "kenttarova", "vehmasmaki", "lindenberg"]
):
    diag = read_diag(site)
    noise = read_noise(site)
    df = diag.merge(noise, on="datetime", how="outer")
    df = df.dropna(subset=["laser_power_percent", "co_std"])
    p = ax.scatter(
        df["laser_power_percent"],
        df["co_std"] ** 2,
        c=mdates.date2num(df.datetime),
        s=1,
    )
    cbar = fig.colorbar(p, ax=ax)
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    cbar.ax.yaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.set_ylabel(r"$\sigma²_{ppol/r^2}$")
    ax.set_xlabel("Laser power (%)")
    ax.grid()
    ax.set_title(site, weight="bold")
axes[0, 0].set_ylim([0, 4e-26])
axes[0, 1].set_ylim([0, 4e-24])
axes[1, 0].set_ylim([0, 9e-26])
# axes[1, 1].set_ylim([0, 3e-13])

# %%
fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
site = "kenttarova"
diag = read_diag(site)
noise = read_noise(site)
df = diag.merge(noise, on="datetime", how="outer")
df = df.dropna(subset=["laser_power_percent", "co_std"])
p = ax.scatter(
    df["laser_power_percent"],
    df["co_std"],
    c=mdates.date2num(df.datetime),
    s=1,
)
cbar = fig.colorbar(p, ax=ax)
cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
cbar.ax.yaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.set_ylabel(r"$\sigma²_{ppol/r^2}$")
ax.set_xlabel("Laser power (%)")
ax.grid()
ax.set_title(site, weight="bold")
ax.set_yscale('log')
# axes[0, 0].set_ylim([0, 4e-26])
# axes[0, 1].set_ylim([0, 4e-24])
# axes[1, 0].set_ylim([0, 9e-26])
# %%
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(df.datetime, df["co_std"])
ax[1].plot(df.datetime, df["integration_time"])

# %%
temp = pd.read_csv(f"/media/viet/CL61/{site}/Noise/20230924.csv")
temp["datetime"] = pd.to_datetime(temp["datetime"])
temp = temp.sort_values("datetime")
temp