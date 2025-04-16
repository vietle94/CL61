import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import string
from cl61.func.calibration_T import temperature_ref, noise_filter, noise_filter_std
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates

# %%
site = "kenttarova"
files = glob.glob(f"/media/viet/CL61/calibration/{site}/merged/*.nc")
df = xr.open_dataset(files[0])
df_mean, df_std = temperature_ref(df)

# %%
fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True, constrained_layout=True)
p = ax[0].pcolormesh(df.time, df.range, df.p_pol.T, norm=LogNorm(vmin=1e-7, vmax=1e-4))
fig.colorbar(p, ax=ax[0])
ax[0].set_ylabel("Range [m]")

ax[1].scatter(
    df_mean.sel(internal_temperature_bins=19).ppol_r,
    df_mean.range,
    s=1,
    label="original",
)
ax[1].scatter(
    df_mean.sel(internal_temperature_bins=19).ppol_ref,
    df_mean.range,
    s=1,
    label="smoothed",
)
ax[2].scatter(
    df_std.sel(internal_temperature_bins=19).ppol_r, df_std.range, s=1, label="original"
)
ax[2].scatter(
    noise_filter_std(df_std.sel(internal_temperature_bins=19).ppol_r),
    df_std.range,
    s=1,
    label="smoothed",
)

ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax[0].xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax[1].set_xlim(-1e-14, 1e-14)
ax[2].set_xlim(2e-14, 6e-14)
for ax_ in ax[1:]:
    ax_.grid()
    ax_.legend(loc="upper right")

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
fig.savefig(
    "/media/viet/CL61/img/cal_example.png",
    dpi=600,
    bbox_inches="tight",
)

# %%
