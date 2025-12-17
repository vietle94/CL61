import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import matplotlib.dates as mdates
import string
from matplotlib.colors import LogNorm

# %%
site = "kenttarova"
files = glob.glob(f"/media/viet/CL61/calibration/{site}/merged/*.nc")
df = xr.open_dataset(files[0])

# %%
df = df.sel(
    time=slice(
        pd.to_datetime("2023-09-26T12:00:00"), pd.to_datetime("2023-09-26T14:01:00")
    )
)
df["ppol_r"] = df.p_pol
df["xpol_r"] = df.x_pol
df_ = df.sel(range=slice(None, 300))
timestep = (
    np.median(df.time.values[1:] - df.time.values[:-1])
    .astype("timedelta64[s]")
    .astype(np.int32)
)  # for fft
fft_ppol = df_.ppol_r.T
fft_ppol = np.fft.fft(fft_ppol.values)
fft_ppol = np.fft.fftshift(fft_ppol, axes=-1)

fft_xpol = df_.xpol_r.T
fft_xpol = np.fft.fft(fft_xpol.values)
fft_xpol = np.fft.fftshift(fft_xpol, axes=-1)

freqx = np.fft.fftfreq(fft_ppol.shape[1], d=timestep)
freqx = np.fft.fftshift(freqx)

fig, ax = plt.subplots(3, 2, figsize=(9, 6), constrained_layout=True, sharex="col")

p = ax[0, 0].pcolormesh(
    df_.time,
    df_.range,
    df_.ppol_r.T,
    shading="nearest",
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
cbar = fig.colorbar(p, ax=ax[0, 0])
cbar.ax.set_ylabel("ppol [a.u.]")


p = ax[1, 0].pcolormesh(
    df_.time,
    df_.range,
    df_.xpol_r.T,
    shading="nearest",
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
cbar = fig.colorbar(p, ax=ax[1, 0])
cbar.ax.set_ylabel("xpol [a.u.]")

ax[2, 0].plot(df_.time, df_.laser_temperature)
ax[2, 0].set_ylabel(r"Laser Temperature [$\degree C$]")
ax[2, 0].grid()

p = ax[0, 1].pcolormesh(
    freqx,
    df_.range,
    np.abs(fft_ppol),
    shading="nearest",
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
cbar = fig.colorbar(p, ax=ax[0, 1])
cbar.ax.set_ylabel("ppol [a.u.]")

p = ax[1, 1].pcolormesh(
    freqx,
    df_.range,
    np.abs(fft_xpol),
    shading="nearest",
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
cbar = fig.colorbar(p, ax=ax[1, 1])
cbar.ax.set_ylabel("xpol [a.u.]")

fft_laser = np.fft.fft(df_.laser_temperature.values) / df_.laser_temperature.size
fft_laser = np.fft.fftshift(fft_laser)
f_signal = np.fft.rfft(df_.laser_temperature.values)
freqx = np.fft.fftfreq(fft_laser.size, d=timestep)
freqx = np.fft.fftshift(freqx)

max_freq = freqx[
    np.max(np.argsort(np.abs(fft_laser), axis=0)[-3:-1])
]  # extract the second highest frequency
ax[2, 1].plot(freqx, np.abs(fft_laser))
ax[2, 1].grid()
ax[2, 1].set_ylabel(r"Laser Temperature [$\degree C$]")
ax[2, 1].axvline(x=max_freq, color="r", linestyle="--")
ax[2, 1].annotate(
    f"Frequency: {max_freq:.2g}Hz\nTime: {1 / max_freq:.2f}s",
    (0.05, 0.8),
    xycoords="axes fraction",
    color="r",
)

for ax_ in ax.flatten()[:4]:
    ax_.set_ylabel("Range [m]")

for ax_ in ax[:, 0]:
    ax_.xaxis.set_major_formatter(mdates.DateFormatter("%Y\n%m-%d\n%H:%M"))
    ax_.xaxis.set_major_locator(mdates.HourLocator(interval=1))


ax[-1, -1].set_xlabel("Frequency [Hz]")
ax[-1, -1].set_ylim(0, 1)

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
fig.savefig("/media/viet/CL61/img/calibration_fft.png", bbox_inches="tight", dpi=600)

# %%
