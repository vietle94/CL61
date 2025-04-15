import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
from cl61.func.calibration_cloud import calibration_etaS
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
import string
import pandas as pd

myFmt = mdates.DateFormatter("%H:%M")

# %%
fig, ax = plt.subplots(
    2, 2, figsize=(9, 6), sharey="row", sharex="col", constrained_layout=True
)

file_dir = "/media/viet/CL61/studycase/lindenberg/20240303/"
df = xr.open_mfdataset(glob.glob(file_dir + "*.nc"))

profile = df.sel(time=slice("2024-03-03T16:30:00", "2024-03-03T17:00:00"))
profile = profile.sel(range=slice(400, 4000))

p = ax[0, 0].pcolormesh(
    profile["time"],
    profile["range"],
    profile["p_pol"].T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
fig.colorbar(p, ax=ax[0, 0], label=r"$\beta$ [m-1 sr-1]")

etaS = calibration_etaS(profile)
c = etaS / 17

ax[0, 0].set_ylabel("Range [m]")
ax[1, 0].set_ylabel("C")
ax[1, 0].plot(c.time, c)

file_dir = "/media/viet/CL61/studycase/lindenberg/20241116/"
df = xr.open_mfdataset(glob.glob(file_dir + "*.nc"))

profile = df.sel(time=slice("2024-11-16T16:00:00", "2024-11-16T18:00:00"))
profile = profile.sel(range=slice(400, 4000))
p = ax[0, 1].pcolormesh(
    profile["time"],
    profile["range"],
    profile["p_pol"].T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
fig.colorbar(p, ax=ax[0, 1], label=r"$\beta$ [m-1 sr-1]")
etaS = calibration_etaS(profile)
c = etaS / 17

ax[1, 1].plot(c.time, c)

for ax_ in ax[1, :]:
    ax_.grid()
    ax_.xaxis.set_major_formatter(myFmt)

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.03,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
ax[0, 0].set_ylim(0, 4000)
ax[1, 0].set_ylim(0, 3)
fig.savefig(
    "/media/viet/CL61/img/studycase_lindenberg_c.png",
    dpi=600,
    bbox_inches="tight",
)
# %%
fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharey=True, sharex='col',
                        constrained_layout=True)

file_dir = "/media/viet/CL61/studycase/lindenberg/20240303/"
df = xr.open_mfdataset(glob.glob(file_dir + "*.nc"))

profile = df.sel(time=slice("2024-03-03T16:30:00", "2024-03-03T17:00:00"))
profile = profile.sel(range=slice(400, 4000))
p = ax[0, 0].pcolormesh(
    profile["time"],
    profile["range"],
    profile["p_pol"].T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
fig.colorbar(p, ax=ax[0, 0], label=r"$\beta$ [m-1 sr-1]")
ax[0, 0].set_ylabel("Range [m]")

etaS = calibration_etaS(profile)
c = etaS / 17
c = np.median(c.values)

ax[0, 2].plot(
    profile["p_pol"].mean(dim="time").values, profile["range"], ".", label="2024-03-03",
    zorder=10
)

p = ax[1, 0].pcolormesh(
    profile["time"],
    profile["range"],
    profile["p_pol"].T*c,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
fig.colorbar(p, ax=ax[1, 0], label=r"$\beta$ [m-1 sr-1]")
ax[1, 0].set_ylabel("Range [m]")
ax[1, 2].plot(
    profile["p_pol"].mean(dim="time").values*c, profile["range"], ".", label="2024-03-03",
    zorder=10
)

file_dir = "/media/viet/CL61/studycase/lindenberg/20241116/"
df = xr.open_mfdataset(glob.glob(file_dir + "*.nc"))

profile = df.sel(time=slice("2024-11-16T16:30:00", "2024-11-16T17:00:00"))
profile = profile.sel(range=slice(400, 4000))
p = ax[0, 1].pcolormesh(
    profile["time"],
    profile["range"],
    profile["p_pol"].T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
fig.colorbar(p, ax=ax[0, 1], label=r"$\beta$ [m-1 sr-1]")

ax[0, 2].plot(
    profile["p_pol"].mean(dim="time").values, profile["range"], ".", label="2024-11-16",
    zorder=1
)
etaS = calibration_etaS(profile)
c = etaS / 17
c = np.median(c.values)

p = ax[1, 1].pcolormesh(
    profile["time"],
    profile["range"],
    profile["p_pol"].T * c,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
fig.colorbar(p, ax=ax[1, 1], label=r"$\beta$ [m-1 sr-1]")
ax[1, 2].plot(
    profile["p_pol"].mean(dim="time").values * c,
    profile["range"],
    ".",
    label="2024-11-16",
    zorder=1,
)


for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.03,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )

for ax_ in ax[:, :-1].flatten():
    ax_.xaxis.set_major_formatter(myFmt)

for ax_ in ax[:, -1].flatten():
    # ax_.set_xscale("log")
    ax_.grid()
    ax_.legend()
    ax_.set_xlim([-0.5e-6, 1e-6])
ax[1, 2].set_xlabel(r"$\beta$ [m-1 sr-1]")
fig.savefig(
    "/media/viet/CL61/img/studycase_lindenberg_corrected.png",
    dpi=600,
    bbox_inches="tight",
)

# %%
file_dir = "/media/viet/CL61/lindenberg/Cloud/"
df = pd.concat(
    [pd.read_csv(f) for f in glob.glob(file_dir + "*.csv")], ignore_index=True
)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

df["c"] = df["c"] * 0.8 * 18.8 / 5
df["mask"] = df["cross_correlation"] > 2e-6
mask = df.set_index("datetime").rolling("1h", center=True).count()["mask"] > 12
mask = mask.reset_index()
df_plot = mask.merge(df)
df_plot = df_plot[df_plot["mask"]]

# %%
df_mean = df_plot.set_index("datetime").resample("7d").median()
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df_plot.datetime, df_plot["c"], ".")
ax.plot(df_mean.index, df_mean["c"])
ax.set_ylabel("C")
ax.grid()
# fig.savefig(
#     "/media/viet/CL61/img/calibration_factor_lindenberg.png",
#     dpi=600,
#     bbox_inches="tight",
# )
# %%
