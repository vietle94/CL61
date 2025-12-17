import string
import xarray as xr
import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm
from cl61.func.noise import noise_detection

# %%
df = xr.open_mfdataset(
    glob.glob("/media/viet/CL61/studycase/kenttarova/20240328/*.nc"),
    preprocess=noise_detection,
)

# %%
df_noise = df.where(df["noise"])
df_noise["p_pol"] = df_noise["p_pol"] / (df["range"] ** 2)
df_noise["x_pol"] = df_noise["x_pol"] / (df["range"] ** 2)
grp_range = df_noise[["p_pol", "x_pol"]].groupby_bins(
    "range", [4000, 6000, 8000, 10000, 12000, 14000]
)
# %%
grp_mean = grp_range.mean(dim=["range"], skipna=False)
grp_std = grp_range.std(dim=["range"], skipna=False)

# %%
fig, ax = plt.subplots(
    4, 2, constrained_layout=True, figsize=(12, 8), sharex=True, sharey="row"
)

p = ax[0, 0].pcolormesh(
    df["time"], df["range"], df["p_pol"].T, norm=LogNorm(vmin=1e-7, vmax=1e-4)
)
cbar = fig.colorbar(p, ax=ax[0, 0])
cbar.ax.set_ylabel("$ppol$ [a.u.]")

p = ax[0, 1].pcolormesh(
    df["time"], df["range"], df["x_pol"].T, norm=LogNorm(vmin=1e-7, vmax=1e-4)
)
cbar = fig.colorbar(p, ax=ax[0, 1])
cbar.ax.set_ylabel("$xpol$ [a.u.]")

p = ax[1, 0].pcolormesh(
    df["time"],
    df["range"],
    df["p_pol"].where(df["noise"]).T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
cbar = fig.colorbar(p, ax=ax[1, 0])
cbar.ax.set_ylabel("$ppol$ [a.u.]")

p = ax[1, 1].pcolormesh(
    df["time"],
    df["range"],
    df["x_pol"].where(df["noise"]).T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
cbar = fig.colorbar(p, ax=ax[1, 1])
cbar.ax.set_ylabel("$xpol$ [a.u.]")
ax[0, 0].set_xlim(df.time.values[0], df.time.values[-1] + pd.Timedelta(minutes=5))

for x in grp_mean.range_bins.values:
    ax[2, 0].plot(
        grp_mean.time,
        grp_mean.sel(range_bins=x).p_pol,
        ".",
        alpha=0.5,
        label=f"{x.left}-{x.right}",
    )
    ax[2, 1].plot(
        grp_mean.time,
        grp_mean.sel(range_bins=x).x_pol,
        ".",
        alpha=0.5,
        label=f"{x.left}-{x.right}",
    )
    ax[3, 0].plot(
        grp_std.time,
        grp_std.sel(range_bins=x).p_pol ** 2,
        ".",
        alpha=0.5,
        label=f"{x.left}-{x.right}",
    )
    ax[3, 1].plot(
        grp_std.time,
        grp_std.sel(range_bins=x).x_pol ** 2,
        ".",
        alpha=0.5,
        label=f"{x.left}-{x.right}",
    )
ax[0, 0].set_ylabel("Range [m]")
ax[1, 0].set_ylabel("Range [m]")
ax[2, 0].set_ylabel(r"$\mu_{ppol/r²}$ [a.u.]")
ax[2, 1].set_ylabel(r"$\mu_{xpol/r²}$ [a.u.]")
ax[3, 0].set_ylabel(r"$\sigma²_{ppol/r²}$ [a.u.]")
ax[3, 1].set_ylabel(r"$\sigma²_{xpol/r²}$ [a.u.]")

handles, labels = ax[3, 1].get_legend_handles_labels()
fig.legend(handles, labels, ncol=6, loc="outside lower center")
for ax_ in ax.flatten()[4:]:
    ax_.grid()
    ax_.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    ax_.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator([0, 6, 12, 18, 24]))

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.1,
        1.1,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
fig.savefig("/media/viet/CL61/img/20240328_bk_noise.png", dpi=400, bbox_inches="tight")
# %%
