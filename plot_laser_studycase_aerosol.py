import xarray as xr
import matplotlib.pyplot as plt
import glob
from matplotlib.colors import LogNorm, SymLogNorm
import string
import matplotlib.dates as mdates

myFmt = mdates.DateFormatter("%H:%M")

# %%
files_before = glob.glob("/media/viet/CL61/studycase/lindenberg/20240307/*.nc")
df_before_ = xr.open_mfdataset(files_before)

df_before = df_before_.sel(time=slice("2024-03-07 20:00:00", "2024-03-08 00:00:00"))

files_after = glob.glob("/media/viet/CL61/studycase/lindenberg/20241201/*.nc")
df_after_ = xr.open_mfdataset(files_after)
df_after = df_after_.sel(time=slice("2024-12-01 20:00:00", "2024-12-02 00:00:00"))

df_before = df_before.sel(range=slice(0, 8000))
df_after = df_after.sel(range=slice(0, 8000))

# %%
profile_before = (df_before["p_pol"]).mean(dim="time")
profile_after = (df_after["p_pol"]).mean(dim="time")

fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, sharey=True)
p = ax[0].pcolormesh(
    df_before["time"],
    df_before["range"],
    df_before["p_pol"].T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
fig.colorbar(p, ax=ax[0], label=r"$\beta$ [m-1 sr-1]")
p = ax[1].pcolormesh(
    df_after["time"],
    df_after["range"],
    df_after["p_pol"].T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
fig.colorbar(p, ax=ax[1], label=r"$\beta$ [m-1 sr-1]")

ax[0].set_ylabel("Range [m]")
ax[0].xaxis.set_major_formatter(myFmt)
ax[0].xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax[1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax[0].set_title("2024-03-07", weight="bold")
ax[1].set_title("2024-12-02", weight="bold")
ax[0].set_xlabel("Time [UTC]")
ax[1].set_xlabel("Time [UTC]")
ax[1].xaxis.set_major_formatter(myFmt)
ax[2].plot(profile_before, profile_before.range, ".", label="2024-03-07", zorder=1)
ax[2].plot(profile_after, profile_after.range, ".", label="2024-12-02", zorder=0)
ax[2].set_xlabel(r"$\beta$ [m-1 sr-1]")
# ax[2].set_xscale("log")
ax[2].set_xlim([-1e-6, 2e-6])
ax[2].legend()
ax[2].grid()

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.03,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
fig.savefig(
    "/media/viet/CL61/img/aerosol_noise_lindenberg.png", dpi=600, bbox_inches="tight"
)
