import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import string
from cl61.func.calibration_T import temperature_ref, noise_filter

# %%
for site, lim in zip(
    ["vehmasmaki", "hyytiala", "kenttarova"],
    [[-1e-14, 1e-14], [-1e-14, 1e-14], [-1e-14, 1e-14]],
):
    files = glob.glob(f"/media/viet/CL61/calibration/{site}/merged/*.nc")
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True, constrained_layout=True)
    for file in files:
        df = xr.open_dataset(file)
        ppol = df["p_pol"].mean(dim="time") / (df["range"]) ** 2
        xpol = df["x_pol"].mean(dim="time") / (df["range"]) ** 2
        ppol_d = noise_filter(ppol)
        xpol_d = noise_filter(xpol)
        ax[0].plot(
            ppol_d,
            ppol.range,
            ".",
            label=df.time.values[0].astype("datetime64[D]").astype(str),
        )
        ax[1].plot(
            xpol_d,
            xpol.range,
            ".",
            label=df.time.values[0].astype("datetime64[D]").astype(str),
        )
        ax[0].set_xlim(lim)
        ax[1].set_xlim([-5e-15, 5e-15])
        df.close()
    ax[0].set_xlabel("ppol/range^2")
    ax[1].set_xlabel("xpol/range^2")
    ax[0].set_ylabel("range")

    for ax_ in ax.flatten():
        ax_.grid()
        ax_.legend()
    fig.savefig(
        f"/media/viet/CL61/img/calibration_noise_{site}.png",
        dpi=600,
        bbox_inches="tight",
    )

# %%
site = "vehmasmaki"
t_range = range(16, 26)
files = glob.glob(f"/media/viet/CL61/calibration/{site}/merged/*.nc")
fig, ax = plt.subplots(
    2, 5, sharex=True, sharey=True, figsize=(12, 6), constrained_layout=True
)
cmap = plt.get_cmap("tab10")
for i, file in enumerate(files):
    df = xr.open_dataset(file)
    df_date = df.time.values[0].astype("datetime64[D]").astype(str)
    print(df_date)
    df["internal_temperature_floor"] = np.floor(df.internal_temperature)
    for ax_, t in zip(ax.flatten(), t_range):
        if t not in df["internal_temperature_floor"].values:
            continue
        df_ = df.where(df.internal_temperature_floor == t)
        df_mean, _ = temperature_ref(df_)
        df_plot = df_mean.sel(internal_temperature_bins=t)
        ax_.plot(df_plot.ppol_ref, df_plot.range, label=df_date, c=cmap(i))
    df.close()

for ax_ in ax.flatten():
    ax_.grid()
    ax_.legend(loc="upper right")
    ax_.set_xlim([-1e-14, 1e-14])

for ax_ in ax[:, 0]:
    ax_.set_ylabel("range")

for ax_ in ax[-1, :]:
    ax_.set_xlabel("ppol/r^2")

for ax_, t in zip(ax.flatten(), t_range):
    ax_.set_title(f"T: {t}°C")

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )

fig.savefig(
    f"/media/viet/CL61/img/calibration_temperature_{site}.png",
    dpi=600,
    bbox_inches="tight",
)
# %%
site = "hyytiala"
t_range = range(13, 15)
files = glob.glob(f"/media/viet/CL61/calibration/{site}/merged/*.nc")
fig, ax = plt.subplots(
    1, 2, sharex=True, sharey=True, figsize=(6, 3), constrained_layout=True
)
cmap = plt.get_cmap("tab10")
for i, file in enumerate(files):
    df = xr.open_dataset(file)
    df_date = df.time.values[0].astype("datetime64[D]").astype(str)
    print(df_date)
    df["internal_temperature_floor"] = np.floor(df.internal_temperature)
    for ax_, t in zip(ax.flatten(), t_range):
        if t not in df["internal_temperature_floor"].values:
            continue
        df_ = df.where(df.internal_temperature_floor == t)
        df_mean, _ = temperature_ref(df_)
        df_plot = df_mean.sel(internal_temperature_bins=t)
        ax_.plot(df_plot.ppol_ref, df_plot.range, label=df_date, c=cmap(i))
    df.close()

for ax_ in ax.flatten():
    ax_.grid()
    ax_.legend(loc="upper right")
    ax_.set_xlim([-1e-14, 1e-14])

ax[0].set_ylabel("range")

for ax_ in ax.flatten():
    ax_.set_xlabel("ppol/r^2")

for ax_, t in zip(ax.flatten(), t_range):
    ax_.set_title(f"T: {t}°C")

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )

fig.savefig(
    f"/media/viet/CL61/img/calibration_temperature_{site}.png",
    dpi=600,
    bbox_inches="tight",
)
# %%
site = "kenttarova"
t_range = range(17, 20)
files = glob.glob(f"/media/viet/CL61/calibration/{site}/merged/*.nc")
fig, ax = plt.subplots(
    1, 3, sharex=True, sharey=True, figsize=(9, 3), constrained_layout=True
)
cmap = plt.get_cmap("tab10")
for i, file in enumerate(files):
    df = xr.open_dataset(file)
    df_date = df.time.values[0].astype("datetime64[D]").astype(str)
    print(df_date)
    df["internal_temperature_floor"] = np.floor(df.internal_temperature)
    for ax_, t in zip(ax.flatten(), t_range):
        if t not in df["internal_temperature_floor"].values:
            continue
        df_ = df.where(df.internal_temperature_floor == t)
        df_mean, _ = temperature_ref(df_)
        df_plot = df_mean.sel(internal_temperature_bins=t)
        ax_.plot(df_plot.ppol_ref, df_plot.range, label=df_date, c=cmap(i))
    df.close()

for ax_ in ax.flatten():
    ax_.grid()
    ax_.legend(loc="upper right")
    ax_.set_xlim([-1e-14, 1e-14])

ax[0].set_ylabel("range")

for ax_ in ax.flatten():
    ax_.set_xlabel("ppol/r^2")

for ax_, t in zip(ax.flatten(), t_range):
    ax_.set_title(f"T: {t}°C")

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )

fig.savefig(
    f"/media/viet/CL61/img/calibration_temperature_{site}.png",
    dpi=600,
    bbox_inches="tight",
)
# %%
