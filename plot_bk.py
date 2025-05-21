import pandas as pd
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
cmap = plt.get_cmap("viridis")
cmap.set_bad("grey")
fig, axes = plt.subplots(
    4, 1, figsize=(9, 8), sharex=True, constrained_layout=True, sharey=True
)
for ax, site in zip(
    axes,
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
    ax1 = ax.twinx()
    p = ax.scatter(
        df["datetime"], (df["co_std"] ** 2) * df["integration"], alpha=0.5, s=1
    )
    ax.set_ylabel(r"$\sigma²_{ppol/r^2} \times t$", color="C0")
    ax.set_yscale("log")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_ylim(top=1e-24)

    p = ax1.scatter(
        diag.datetime, diag["laser_power_percent"], alpha=0.5, s=1, color="C1"
    )
    ax1.set_ylabel("Laser power (%)", color="C1")
    ax1.tick_params(axis="y", labelcolor="C1")
    ax1.grid()
    ax.grid()

axes[0].axvspan("2022-06-15", "2023-04-27", color="C2", alpha=0.2, label="1.1.10")
axes[0].axvspan("2023-04-28", "2025-01-01", color="C3", alpha=0.2, label="1.2.7")

axes[1].axvspan("2022-11-21", "2023-11-22", color="C2", alpha=0.2, label="1.1.10")
axes[1].axvspan("2023-11-23", "2025-01-01", color="C3", alpha=0.2, label="1.2.7")

axes[2].axvspan("2023-06-21", "2025-01-01", color="C3", alpha=0.2, label="1.2.7")

axes[3].axvspan("2024-03-01", "2025-01-01", color="C2", alpha=0.2, label="1.1.10")

for n, ax_ in enumerate(axes.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
    ax_.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_.xaxis.set_major_locator(mdates.MonthLocator(12))
    ax_.xaxis.set_minor_locator(mdates.MonthLocator(6))
    ax_.legend(loc="lower left")

fig.savefig("/media/viet/CL61/img/bk_ts.png", dpi=600, bbox_inches="tight")


# %%
def read_diag(site, time=("23:00", "01:00")):
    diag = pd.DataFrame({})
    for file in glob.glob(f"/media/viet/CL61/{site}/Diag/*.csv"):
        df = pd.read_csv(file)
        try:
            df = df[["datetime", "laser_power_percent", "internal_temperature"]]
        except KeyError:
            continue
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[df["datetime"] > "2000-01-01"]
        if site == "lindenberg":
            df["datetime"] = (
                df["datetime"]
                .dt.tz_localize("UTC")
                .dt.tz_convert("Europe/Berlin")
                .dt.tz_localize(None)
            )
        else:
            df["datetime"] = (
                df["datetime"]
                .dt.tz_localize("UTC")
                .dt.tz_convert("Europe/Helsinki")
                .dt.tz_localize(None)
            )
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


def read_noise(site, time=("23:00", "01:00")):
    noise = pd.DataFrame({})
    for file in glob.glob(f"/media/viet/CL61/{site}/Noise/*.csv"):
        df = pd.read_csv(file)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[df["datetime"] > "2000-01-01"]
        if site == "lindenberg":
            df["datetime"] = (
                df["datetime"]
                .dt.tz_localize("UTC")
                .dt.tz_convert("Europe/Berlin")
                .dt.tz_localize(None)
            )
        else:
            df["datetime"] = (
                df["datetime"]
                .dt.tz_localize("UTC")
                .dt.tz_convert("Europe/Helsinki")
                .dt.tz_localize(None)
            )
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
    axes.flatten(), ["vehmasmaki", "hyytiala", "kenttarova", "lindenberg"]
):
    integration_files = glob.glob(f"/media/viet/CL61/{site}/Integration/*.csv")
    df_integration = pd.concat(
        [pd.read_csv(x) for x in integration_files], ignore_index=True
    )
    df_integration["datetime"] = pd.to_datetime(
        df_integration["datetime"], format="mixed"
    )
    df_integration["date"] = df_integration["datetime"].dt.date
    df_integration.drop(columns=["datetime"], inplace=True)

    diag = read_diag(site, time=("01:00", "23:00"))
    noise = read_noise(site, time=("01:00", "23:00"))
    df = diag.merge(noise, on="datetime", how="outer")
    df = df.dropna(subset=["laser_power_percent", "co_std"])
    df["date"] = df.datetime.dt.date
    df = df.merge(df_integration, on="date", how="outer")
    ax.scatter(
        df["laser_power_percent"],
        (df["co_std"] ** 2) * df["integration"],
        c="grey",
        alpha=0.5,
        s=1,
        label="All day",
    )

    diag = read_diag(site)
    noise = read_noise(site)
    df = diag.merge(noise, on="datetime", how="outer")
    df = df.dropna(subset=["laser_power_percent", "co_std"])
    df["date"] = df.datetime.dt.date
    df = df.merge(df_integration, on="date", how="outer")
    ax.scatter(
        df["laser_power_percent"],
        (df["co_std"] ** 2) * df["integration"],
        alpha=0.5,
        s=1,
        label="23:00 - 01:00",
    )
    ax.grid()
    ax.set_yscale("log")
    ax.legend(loc="lower left")

for n, ax_ in enumerate(axes.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
axes[1, 0].set_xlabel("Laser power (%)")
axes[1, 1].set_xlabel("Laser power (%)")
axes[0, 0].set_ylabel(r"$\sigma²_{ppol/r^2} \times t_{integration}$")
axes[1, 0].set_ylabel(r"$\sigma²_{ppol/r^2} \times t_{integration}$")
fig.savefig("/media/viet/CL61/img/bk_laser.png", dpi=600, bbox_inches="tight")
