import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.dates as mdates
import locale
import string

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


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
            .resample("1h")
            .mean()
            .reset_index()
        )
        diag = pd.concat([diag, df_1h])
    diag = diag.reset_index(drop=True)
    return diag


def read_noise(site, time=("22:00", "23:59")):
    noise = pd.DataFrame({})
    for file in glob.glob(f"/media/viet/CL61/{site}/Noise/*.csv"):
        df = pd.read_csv(file)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[df["datetime"] > "2000-01-01"]
        df = df.groupby("range").get_group("(6000, 8000]")
        df_1h = (
            df.set_index("datetime")
            .between_time(*time)
            .resample("1h")
            .mean(numeric_only=True)
            .reset_index()
        )
        noise = pd.concat([noise, df_1h])
    noise = noise.reset_index(drop=True)
    return noise


# %%
fig, axes = plt.subplots(
    2, 2, figsize=(9, 6), sharex=True, constrained_layout=True, sharey=True
)
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
    ax.set_yscale("log")
    ax.set_title(site, weight="bold")
    ax.yaxis.set_tick_params(labelbottom=True)

for n, ax_ in enumerate(axes.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
ax.set_ylim([0, 1e-24])
ax.yaxis.set_tick_params(labelbottom=True)
fig.savefig("/media/viet/CL61/img/laser_power.png", dpi=600, bbox_inches="tight")

# %%
fig, ax = plt.subplots(
    figsize=(5, 3), sharex=True, sharey=True, constrained_layout=True
)
site = "lindenberg"

integration_files = glob.glob(f"/media/viet/CL61/{site}/Integration/*.csv")
df_integration = pd.concat(
    [pd.read_csv(x) for x in integration_files], ignore_index=True
)
df_integration["datetime"] = pd.to_datetime(df_integration["datetime"], format="mixed")
df_integration["date"] = df_integration["datetime"].dt.date
df_integration.drop(columns=["datetime"], inplace=True)

diag = read_diag(site, time=("00:00", "20:00"))
noise = read_noise(site, time=("00:00", "20:00"))
df = diag.merge(noise, on="datetime", how="outer")
df = df.dropna(subset=["laser_power_percent", "co_std"])
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
cbar.ax.yaxis.set_minor_locator(mdates.MonthLocator(interval=1))

ax.set_axisbelow(True)
ax.set_ylabel(r"$\sigma²_{ppol/r^2} \times t_{integration}$")
ax.grid(linestyle="--")
ax.set_xlabel("Laser power (%)")

diag = read_diag(site)
noise = read_noise(site)
df = diag.merge(noise, on="datetime", how="outer")
df = df.dropna(subset=["laser_power_percent", "co_std"])
df["date"] = df.datetime.dt.date
df = df.merge(df_integration, on="date", how="outer")

p = ax.scatter(
    df["laser_power_percent"], (df["co_std"] ** 2) * df["integration"], c="grey", s=1
)
ax.set_yscale("log")
fig.savefig(
    "/media/viet/CL61/img/laser_power_lindenberg.png", dpi=600, bbox_inches="tight"
)
# %%
cmap = plt.get_cmap("viridis")
cmap.set_bad("grey")
fig, axes = plt.subplots(
    4, 1, figsize=(8, 8), sharex=True, constrained_layout=True, sharey=True
)
for ax, site in zip(
    axes.flatten(),
    ["vehmasmaki", "hyytiala", "kenttarova", "lindenberg"],
):
    diag = read_diag(site)
    noise = read_noise(site)
    df = noise.merge(diag, on="datetime", how="left")
    # df = df.dropna(subset=["laser_power_percent", "co_std"])
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
        df["datetime"],
        (df["co_std"] ** 2) * df["integration"],
        c=df["laser_power_percent"],
        vmin=10,
        vmax=100,
        plotnonfinite=True,
        cmap=cmap,
        s=1,
    )
    ax.set_ylabel(r"$\sigma²_{ppol/r^2} \times t$")
    ax.grid()
    ax.set_yscale("log")
    ax.yaxis.set_tick_params(labelbottom=True)
cbar = fig.colorbar(p, ax=axes, label="Laser power (%)")
ax_flat = axes.flatten()
ax_flat[0].axvspan("2022-06-15", "2023-04-27", color="C0", alpha=0.2, label="1.1.10")
ax_flat[0].axvspan("2023-04-28", "2025-01-01", color="C1", alpha=0.2, label="1.2.7")
ax_flat[1].axvspan("2022-11-21", "2023-11-22", color="C0", alpha=0.2, label="1.1.10")
ax_flat[1].axvspan("2023-11-23", "2025-01-01", color="C1", alpha=0.2, label="1.1.10")
ax_flat[2].axvspan("2023-06-21", "2025-01-01", color="C1", alpha=0.2, label="1.1.7")
ax_flat[3].axvspan("2024-03-01", "2025-01-01", color="C0", alpha=0.2, label="1.1.10")

for n, ax_ in enumerate(axes.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
    ax_.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_.xaxis.set_major_locator(mdates.MonthLocator([6, 12]))
    ax_.legend(loc="upper left")
ax.set_ylim([0, 1e-24])
fig.savefig("/media/viet/CL61/img/laser_power_ts.png", dpi=600, bbox_inches="tight")

# %%
fig, axes = plt.subplots(
    2, 2, figsize=(9, 6), sharex=True, constrained_layout=True, sharey=True
)
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
        c=df["integration"],
        s=1,
    )
    cbar = fig.colorbar(p, ax=ax, label="Integration time (s)")
    ax.set_ylabel(r"$\sigma²_{ppol/r^2} \times t_{integration}$")
    ax.set_xlabel("Laser power (%)")
    ax.grid()
    ax.set_yscale("log")
    ax.set_title(site, weight="bold")
    ax.yaxis.set_tick_params(labelbottom=True)

for n, ax_ in enumerate(axes.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
ax.set_ylim([0, 1e-24])
ax.yaxis.set_tick_params(labelbottom=True)
fig.savefig("/media/viet/CL61/img/laser_power_t.png", dpi=600, bbox_inches="tight")
# %%
