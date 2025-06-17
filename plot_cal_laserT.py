import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm, SymLogNorm

# %%
site = "kenttarova"
files = glob.glob(f"/media/viet/CL61/calibration/{site}/merged/*.nc")
df = xr.open_dataset(files[0])

# %%
df["ppol_r"] = df.p_pol / (df.range**2)
df["xpol_r"] = df.x_pol / (df.range**2)
df_ = df.sel(range=slice(0, 300))
timestep = (
    np.median(df.time.values[1:] - df.time.values[:-1])
    .astype("timedelta64[s]")
    .astype(np.int32)
)  # for fft
fft_ppol = df_.ppol_r.T
fft_ppol = np.fft.fft(fft_ppol.values) / fft_ppol.size
fft_ppol = np.fft.fftshift(fft_ppol, axes=-1)

fft_xpol = df_.xpol_r.T
fft_xpol = np.fft.fft(fft_xpol.values) / fft_xpol.size
fft_xpol = np.fft.fftshift(fft_xpol, axes=-1)

freqx = np.fft.fftfreq(fft_ppol.shape[1], d=timestep)
freqx = np.fft.fftshift(freqx)

fig, ax = plt.subplots(3, 2, figsize=(9, 6), constrained_layout=True, sharex="col")

p = ax[0, 0].pcolormesh(
    df_.time,
    df_.range,
    df_.ppol_r.T,
    shading="nearest",
    norm=SymLogNorm(linthresh=1e-15, vmin=-1e-9, vmax=1e-9),
    cmap="RdBu",
)
cbar = fig.colorbar(p, ax=ax[0, 0])
cbar.ax.set_ylabel("ppol")


p = ax[1, 0].pcolormesh(
    df_.time,
    df_.range,
    df_.xpol_r.T,
    shading="nearest",
    norm=SymLogNorm(linthresh=1e-15, vmin=-1e-9, vmax=1e-9),
    cmap="RdBu",
)
cbar = fig.colorbar(p, ax=ax[1, 0])
cbar.ax.set_ylabel("xpol")

ax[2, 0].plot(df_.time, df_.laser_temperature)
ax[2, 0].set_ylabel("Laser T")
ax[2, 0].grid()

p = ax[0, 1].pcolormesh(
    freqx,
    df_.range,
    np.abs(fft_ppol),
    shading="nearest",
    norm=LogNorm(vmin=1e-15, vmax=1e-12),
)
cbar = fig.colorbar(p, ax=ax[0, 1])
cbar.ax.set_ylabel("FFT_Amplitude/2")

p = ax[1, 1].pcolormesh(
    freqx,
    df_.range,
    np.abs(fft_xpol),
    shading="nearest",
    norm=LogNorm(vmin=1e-15, vmax=1e-12),
)
cbar = fig.colorbar(p, ax=ax[1, 1])
cbar.ax.set_ylabel("FFT_Amplitude/2")

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
ax[2, 1].set_ylabel("FFT Amplitude/2")
ax[2, 1].axvline(x=max_freq, color="r", linestyle="--")
ax[2, 1].annotate(
    f"Frequency: {max_freq:.2g}Hz\nTime: {1 / max_freq:.2f}s",
    (0.05, 0.8),
    xycoords="axes fraction",
    color="r",
)

for ax_ in ax[:, 0]:
    ax_.xaxis.set_major_formatter(mdates.DateFormatter("%Y\n%m-%d\n%H:%M"))
#     ax_.xaxis.set_major_locator(mdates.HourLocator(interval = 3))


ax[-1, -1].set_xlabel("Frequency (Hz)")
ax[-1, -1].set_ylim(0, 1)
ax[0, 0].set_xlim(df_.time.values.min(), df_.time.values.min() + pd.Timedelta("1h"))


# %%
fft_laser = np.fft.fft(df_.laser_temperature.values)
freqx = np.fft.fftfreq(fft_laser.size, d=timestep)

cut_f_signal = fft_laser.copy()
cut_f_signal[(freqx > 0.0065) & (freqx < 0.0095)] = 0
cut_f_signal[(freqx > -0.0095) & (freqx < -0.0065)] = 0
cut_signal = np.fft.ifft(cut_f_signal)

fft_laser = np.fft.fftshift(fft_laser)
freqx = np.fft.fftshift(freqx)


fig, ax = plt.subplots(2, 1, figsize=(16, 9))
ax[0].plot(df_.time, df_.laser_temperature)
ax[1].plot(df_.time, cut_signal)
for ax_ in ax:
    ax_.xaxis.set_major_formatter(mdates.DateFormatter("%Y\n%m-%d\n%H:%M"))
    ax_.grid()
# %%
df_ = df.sel(range=slice(0, 100))
fft_ppol = df_.ppol_r.T
fft_ppol = np.fft.fft(fft_ppol.values)
fft_xpol = df_.xpol_r.T
fft_xpol = np.fft.fft(fft_xpol.values)
freqx = np.fft.fftfreq(fft_ppol.shape[1], d=timestep)
mask = (np.abs(freqx) < 0.0065) ^ (np.abs(freqx) < 0.009)  # filter frequencies range
ppol_signal = fft_ppol.copy()
ppol_signal[:, mask] = 0
ppol_signal = np.fft.ifft(ppol_signal)

xpol_signal = fft_xpol.copy()
xpol_signal[:, mask] = 0
xpol_signal = np.fft.ifft(xpol_signal)

fig, ax = plt.subplots(2, 2, figsize=(9, 4), constrained_layout=True, sharex=True)

p = ax[0, 0].pcolormesh(
    df_.time,
    df_.range,
    df_.ppol_r.T,
    shading="nearest",
    norm=SymLogNorm(linthresh=1e-15, vmin=-1e-9, vmax=1e-9),
    cmap="RdBu",
)
cbar = fig.colorbar(p, ax=ax[0, 0])
cbar.ax.set_ylabel("ppol")


p = ax[1, 0].pcolormesh(
    df_.time,
    df_.range,
    df_.xpol_r.T,
    shading="nearest",
    norm=SymLogNorm(linthresh=1e-15, vmin=-1e-9, vmax=1e-9),
    cmap="RdBu",
)
cbar = fig.colorbar(p, ax=ax[1, 0])
cbar.ax.set_ylabel("xpol")

p = ax[0, 1].pcolormesh(
    df_.time,
    df_.range,
    ppol_signal.real,
    shading="nearest",
    norm=SymLogNorm(linthresh=1e-15, vmin=-1e-9, vmax=1e-9),
    cmap="RdBu",
)
cbar = fig.colorbar(p, ax=ax[0, 1])
cbar.ax.set_ylabel("ppol")

p = ax[1, 1].pcolormesh(
    df_.time,
    df_.range,
    xpol_signal.real,
    shading="nearest",
    norm=SymLogNorm(linthresh=1e-15, vmin=-1e-9, vmax=1e-9),
    cmap="RdBu",
)
cbar = fig.colorbar(p, ax=ax[1, 1])
cbar.ax.set_ylabel("xpol")
ax[0, 0].set_xlim(df_.time.values.min(), df_.time.values.min() + pd.Timedelta("1h"))
for ax_ in ax.flatten():
    ax_.xaxis.set_major_formatter(mdates.DateFormatter("%Y\n%m-%d\n%H:%M"))

# %%
fig, ax = plt.subplots(
    2, 2, figsize=(16, 9), sharey="col", sharex=True, constrained_layout=True
)
ax[0, 0].plot(df_.time, df_.xpol_r.T.values[5, :])
ax[1, 0].plot(df_.time, xpol_signal.real[5, :])

ax[0, 1].plot(df_.time, df_.ppol_r.T.values[5, :])
ax[1, 1].plot(df_.time, ppol_signal.real[5, :])

for ax_ in ax.flatten():
    ax_.grid()
    ax_.xaxis.set_major_formatter(mdates.DateFormatter("%Y\n%m-%d\n%H:%M"))
