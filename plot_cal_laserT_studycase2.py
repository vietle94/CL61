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
from cl61.func.utils import smooth

# %%
df = xr.open_mfdataset(glob.glob("/media/viet/CL61/studycase/kenttarova/20240614/*.nc"))
noise = df.sel(
    range=slice(None, 300),
    time=slice(
        pd.to_datetime("2024-06-14T120000"), pd.to_datetime("2024-06-14T143000")
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

# f_noise = fft.stft(noise_centered.values, axis=0)
f_noise = fft.stft(noise.values, axis=0)
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
# noise_filtered = noise_filtered + noise_mean.values
noise_filtered = noise_filtered

# %%
df = xr.open_mfdataset(glob.glob("/media/viet/CL61/studycase/kenttarova/20240305/*.nc"))
df = df.sel(range=slice(None, 300),
            time=slice(pd.to_datetime("2024-03-05T110000"), pd.to_datetime("2024-03-05T130000")))
# %%
signal = df.p_pol
signal_mean = signal.mean(dim="time")
# signal_centered = signal - signal_mean
signal_centered = signal
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
# ppol_filtered = signal_filtered + signal_mean.values
ppol_filtered = signal_filtered

signal = df.x_pol
signal_mean = signal.mean(dim="time")
# signal_centered = signal - signal_mean
signal_centered = signal
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
# xpol_filtered = signal_filtered + signal_mean.values
xpol_filtered = signal_filtered

# %%
fig, ax = plt.subplots(
    4, 2, sharex="col", sharey=True, figsize=(12, 9), constrained_layout=True
)
p = ax[0, 0].pcolormesh(
    signal.time.values,
    signal.range.values,
    df.p_pol.T,
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
    df.x_pol.T,
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
# fig.savefig(
#     "/media/viet/CL61/img/calibration_fft_studycase.png", bbox_inches="tight", dpi=600
# )

# %%
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
# ax.plot(signal.time, ppol_filtered[:, 4], '.', label="ppol_fft")
ax.plot(
    signal.time,
    df.sel(
        time=slice(
            pd.to_datetime("2024-03-05T110000"), pd.to_datetime("2024-03-05T130000")
        ),
    ).isel(range=4).x_pol,
    label="ppol_original",
)
ax.plot(
    signal.time,
    smooth(df.sel(
        time=slice(
            pd.to_datetime("2024-03-05T110000"), pd.to_datetime("2024-03-05T130000")
        ),
    )
    .isel(range=4)
    .x_pol, 30),
    label="ppol_avg",
)
# ax.set_ylim(4e-7, 7.5e-7)
ax.legend()

# %%
df = df.sel(range=slice(None, 3000))

# %%
fig, ax = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True, sharex=True, sharey=True)
# p = ax[0].pcolormesh(df.time, df.range, df.p_pol.T, norm=LogNorm(vmin=1e-7, vmax=1e-4), shading='auto')
# fig.colorbar(p, ax=ax[0])
p = ax[1].pcolormesh(
    df.time, df.range[:-1], np.diff(df.p_pol).T, vmin=-1e-8, vmax=1e-8, shading="auto", cmap="RdBu_r"
)
fig.colorbar(p, ax=ax[1])
# ax[0].set_ylim(0, 500)
# %%
df_diag = xr.open_mfdataset(glob.glob("/media/viet/CL61/studycase/hyytiala/20240805/*.nc"),
                            group="monitoring")
# %%
for x in list(df_diag.keys()):
    fig, ax = plt.subplots()
    ax.plot(df_diag.time, df_diag[x])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title(x)


# %%
df = xr.open_dataset("/media/viet/CL61/calibration/vehmasmaki/merged/20241127.nc")

# %%
fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(12, 6))
p = ax[0].pcolormesh(df.time, df.range, df.p_pol.T, norm=LogNorm(1e-7, 1e-4))
fig.colorbar(p, ax=ax[0])
ax[0].set_ylim(0, 300)
ax[1].plot(df.time, df.internal_temperature, label="internal temperature")
ax[0].set_xlim(
    pd.to_datetime("2024-11-27T12:00:00"), pd.to_datetime("2024-11-27T18:59:59")
)
# %%
fig, ax = plt.subplots()
ax.plot(df.time, df.p_pol.isel(range=4), label="ppol")
ax.set_xlim(
    pd.to_datetime("2024-11-27T12:00:00"), pd.to_datetime("2024-11-27T18:59:59")
)
# %%
df = xr.open_mfdataset(glob.glob("/media/viet/CL61/studycase/hyytiala/20240805/*.nc"))
df = df.sel(range=slice(None, 3000))

# %%
df = df.sel(time=slice(pd.to_datetime("2024-08-05T090000"), pd.to_datetime("2024-08-05T150000")))

# %%
fig, ax = plt.subplots(constrained_layout=True, figsize=(12, 6))
p = ax.pcolormesh(df.time, df.range, df.p_pol.T, norm=LogNorm(1e-5, 1e-8))
fig.colorbar(p, ax=ax)
ax.set_ylim(0, 300)
ax.set_xlim(pd.to_datetime("2024-08-05T10:00:00"), pd.to_datetime("2024-08-05T11:00:00"))
# %%
fig, ax = plt.subplots()
ax.plot(df.time, df.p_pol.isel(range=4), label="ppol")
# ax.set_ylim(0, 1e-7)
ax.set_xlim(pd.to_datetime("2024-08-05T10:00:00"), pd.to_datetime("2024-08-05T11:00:00"))
# %%
df = xr.open_mfdataset(glob"/media/viet/CL61/studycase/hyytiala/*.nc")

# %%
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
p = ax.pcolormesh(df.time, df.range, df.p_pol.T, norm=LogNorm(1e-7, 1e-4))
fig.colorbar(p, ax=ax)
ax.set_ylim(0, 300)
# %%
