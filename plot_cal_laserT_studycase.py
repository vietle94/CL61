import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.dates as mdates
import string
from scipy.ndimage import median_filter
from scipy.stats import iqr
from scipy.signal import ShortTimeFFT, get_window
from matplotlib.colors import LogNorm
import pandas as pd

# %%
df = xr.open_mfdataset(glob.glob("/media/viet/CL61/studycase/kenttarova/20230926/*.nc"))
noise = df.sel(
    range=slice(None, 300),
    time=slice(
        pd.to_datetime("2023-09-26T111500"), pd.to_datetime("2023-09-26T161000")
    ),
).p_pol
noise_mean = noise.mean(dim="time")
noise_centered = noise - noise_mean
# %%
fft = ShortTimeFFT(
    win=get_window("hamm", 256),
    hop=128,
    fs=10,
)

f_noise = fft.stft(noise_centered.values, axis=0)
mag_noise = np.abs(f_noise)
phase_noise = np.angle(f_noise)

# %%
mag_noise_median = median_filter(mag_noise, size=(21, 1, 1))
residual = mag_noise - mag_noise_median
iqr_value = iqr(residual, axis=0)
# Mark as outliers using IQR rule (e.g., 1.5x IQR)
outlier_mask = residual > (1.5 * iqr_value)
outlier_mask = np.max(outlier_mask, axis=2)
outlier_mask_noise = np.repeat(outlier_mask[:, :, np.newaxis], f_noise.shape[2], axis=2)

mag_noise_filtered = np.where(outlier_mask_noise, mag_noise_median, mag_noise)
mag_noise_filtered[:5, :, :] = mag_noise[:5, :, :]  # keep the first frequency unchanged
f_noise_filtered = mag_noise_filtered * np.exp(1j * phase_noise)
noise_filtered = np.real(
    fft.istft(f_noise_filtered, k1=noise.shape[0], f_axis=0, t_axis=2)
)
noise_filtered = noise_filtered + noise_mean.values

# %%
df = xr.open_mfdataset(glob.glob("/media/viet/CL61/studycase/kenttarova/20230926/*.nc"))

# %%
signal = df.sel(range=slice(None, 300)).p_pol
signal_mean = signal.mean(dim="time")
signal_centered = signal - signal_mean
f_signal = fft.stft(signal_centered.values, axis=0)
mag_signal = np.abs(f_signal)
phase_signal = np.angle(f_signal)

mag_signal_median = median_filter(mag_signal, size=(21, 1, 1))
outlier_mask_signal = np.repeat(
    outlier_mask[:, :, np.newaxis], f_signal.shape[2], axis=2
)
mag_signal_filtered = np.where(outlier_mask_signal, mag_signal_median, mag_signal)
mag_signal_filtered[:5, :, :] = mag_signal[
    :5, :, :
]  # keep the first frequency unchanged
f_signal_filtered = mag_signal_filtered * np.exp(1j * phase_signal)
signal_filtered = np.real(
    fft.istft(f_signal_filtered, k1=signal.shape[0], f_axis=0, t_axis=2)
)
ppol_filtered = signal_filtered + signal_mean.values

signal = df.sel(range=slice(None, 300)).x_pol
signal_mean = signal.mean(dim="time")
signal_centered = signal - signal_mean
f_signal = fft.stft(signal_centered.values, axis=0)
mag_signal = np.abs(f_signal)
phase_signal = np.angle(f_signal)

mag_signal_median = median_filter(mag_signal, size=(21, 1, 1))
outlier_mask_signal = np.repeat(
    outlier_mask[:, :, np.newaxis], f_signal.shape[2], axis=2
)
mag_signal_filtered = np.where(outlier_mask_signal, mag_signal_median, mag_signal)
mag_signal_filtered[:5, :, :] = mag_signal[
    :5, :, :
]  # keep the first frequency unchanged
f_signal_filtered = mag_signal_filtered * np.exp(1j * phase_signal)
signal_filtered = np.real(
    fft.istft(f_signal_filtered, k1=signal.shape[0], f_axis=0, t_axis=2)
)
xpol_filtered = signal_filtered + signal_mean.values

# %%
fig, ax = plt.subplots(
    4, 2, sharex="col", sharey=True, figsize=(12, 9), constrained_layout=True
)
p = ax[0, 0].pcolormesh(
    signal.time.values,
    signal.range.values,
    df.sel(range=slice(None, 300)).p_pol.T,
    shading="auto",
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
cbar = fig.colorbar(p, ax=ax[0, 0])
cbar.ax.set_ylabel("ppol")

p = ax[1, 0].pcolormesh(
    signal.time.values,
    signal.range.values,
    ppol_filtered.T,
    shading="auto",
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
cbar = fig.colorbar(p, ax=ax[1, 0])
cbar.ax.set_ylabel("ppol")

p = ax[2, 0].pcolormesh(
    signal.time.values,
    signal.range.values,
    df.sel(range=slice(None, 300)).x_pol.T,
    shading="auto",
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
cbar = fig.colorbar(p, ax=ax[2, 0])
cbar.ax.set_ylabel("xpol")

p = ax[3, 0].pcolormesh(
    signal.time.values,
    signal.range.values,
    xpol_filtered.T,
    shading="auto",
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
cbar = fig.colorbar(p, ax=ax[3, 0])
cbar.ax.set_ylabel("xpol")

df = xr.open_mfdataset(glob.glob("/media/viet/CL61/studycase/kenttarova/20231002/*.nc"))

signal = df.sel(range=slice(None, 300)).p_pol
signal_mean = signal.mean(dim="time")
signal_centered = signal - signal_mean
f_signal = fft.stft(signal_centered.values, axis=0)
mag_signal = np.abs(f_signal)
phase_signal = np.angle(f_signal)

mag_signal_median = median_filter(mag_signal, size=(21, 1, 1))
outlier_mask_signal = np.repeat(
    outlier_mask[:, :, np.newaxis], f_signal.shape[2], axis=2
)
mag_signal_filtered = np.where(outlier_mask_signal, mag_signal_median, mag_signal)
mag_signal_filtered[:5, :, :] = mag_signal[
    :5, :, :
]  # keep the first frequency unchanged
f_signal_filtered = mag_signal_filtered * np.exp(1j * phase_signal)
signal_filtered = np.real(
    fft.istft(f_signal_filtered, k1=signal.shape[0], f_axis=0, t_axis=2)
)
ppol_filtered = signal_filtered + signal_mean.values

signal = df.sel(range=slice(None, 300)).x_pol
signal_mean = signal.mean(dim="time")
signal_centered = signal - signal_mean
f_signal = fft.stft(signal_centered.values, axis=0)
mag_signal = np.abs(f_signal)
phase_signal = np.angle(f_signal)

mag_signal_median = median_filter(mag_signal, size=(21, 1, 1))
outlier_mask_signal = np.repeat(
    outlier_mask[:, :, np.newaxis], f_signal.shape[2], axis=2
)
mag_signal_filtered = np.where(outlier_mask_signal, mag_signal_median, mag_signal)
mag_signal_filtered[:5, :, :] = mag_signal[
    :5, :, :
]  # keep the first frequency unchanged
f_signal_filtered = mag_signal_filtered * np.exp(1j * phase_signal)
signal_filtered = np.real(
    fft.istft(f_signal_filtered, k1=signal.shape[0], f_axis=0, t_axis=2)
)
xpol_filtered = signal_filtered + signal_mean.values

p = ax[0, 1].pcolormesh(
    signal.time.values,
    signal.range.values,
    df.sel(range=slice(None, 300)).p_pol.T,
    shading="auto",
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
cbar = fig.colorbar(p, ax=ax[0, 1])
cbar.ax.set_ylabel("ppol")

p = ax[1, 1].pcolormesh(
    signal.time.values,
    signal.range.values,
    ppol_filtered.T,
    shading="auto",
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
cbar = fig.colorbar(p, ax=ax[1, 1])
cbar.ax.set_ylabel("ppol")

p = ax[2, 1].pcolormesh(
    signal.time.values,
    signal.range.values,
    df.sel(range=slice(None, 300)).x_pol.T,
    shading="auto",
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
cbar = fig.colorbar(p, ax=ax[2, 1])
cbar.ax.set_ylabel("xpol")

p = ax[3, 1].pcolormesh(
    signal.time.values,
    signal.range.values,
    xpol_filtered.T,
    shading="auto",
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
cbar = fig.colorbar(p, ax=ax[3, 1])
cbar.ax.set_ylabel("xpol")


ax[0, 0].set_xlim(
    pd.to_datetime("2023-09-26 10:00"), pd.to_datetime("2023-09-26 12:00")
)
ax[0, 1].set_xlim(
    pd.to_datetime("2023-10-02 10:00"), pd.to_datetime("2023-10-02 12:00")
)
for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
    ax_.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax_.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
fig.savefig(
    "/media/viet/CL61/img/calibration_fft_studycase.png", bbox_inches="tight", dpi=600
)
