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
from cl61.func.near_range import analyze_noise, denoise_fft

# %%
df = xr.open_mfdataset(glob.glob("/media/viet/CL61/studycase/kenttarova/20230926/*.nc"))
noise = df.sel(
    range=slice(None, 300),
    time=slice(
        pd.to_datetime("2023-09-26T111500"), pd.to_datetime("2023-09-26T161000")
    ),
)
noise_100 = noise.sel(range=slice(None, 100))
# ppol
ppol_noise_filtered, outlier_mask = analyze_noise(
    noise_100.p_pol
)
noise_100["ppol_filtered"] = (("time", "range"), ppol_noise_filtered)
noise["ppol_filtered"] = xr.concat(
    [noise_100["ppol_filtered"], noise.where(noise.range > 100, drop=True)["p_pol"]],
    dim="range",
)
ppol_noise_profile = noise.sel(
    time=slice(pd.to_datetime("2023-09-26T120000"), pd.to_datetime("2023-09-26T150000"))
)["ppol_filtered"].mean(dim="time")

# xpol
xpol_noise_filtered, outlier_mask = analyze_noise(noise_100.x_pol)
noise_100["xpol_filtered"] = (("time", "range"), xpol_noise_filtered)
noise["xpol_filtered"] = xr.concat(
    [noise_100["xpol_filtered"], noise.where(noise.range > 100, drop=True)["p_pol"]],
    dim="range",
)
xpol_noise_profile = noise.sel(
    time=slice(pd.to_datetime("2023-09-26T120000"), pd.to_datetime("2023-09-26T150000"))
)["xpol_filtered"].mean(dim="time")


# %% Get the f_c from reference profiles
df_fc = df.sel(
    range=slice(None, 200),
    time=slice(
        pd.to_datetime("2023-09-26T080000"), pd.to_datetime("2023-09-26T110000")
    ),
)
df_fc_100 = df_fc.sel(range=slice(None, 100))

# ppol
df_fc_100["ppol_filtered"] = (
    ("time", "range"),
    denoise_fft(df_fc_100.p_pol, outlier_mask),
)
df_fc["ppol_filtered"] = xr.concat(
    [df_fc_100["ppol_filtered"], df_fc.where(noise.range > 100, drop=True)["p_pol"]],
    dim="range",
)

denoised = (
    df_fc.sel(
        time=slice(
            pd.to_datetime("2023-09-26T100000"), pd.to_datetime("2023-09-26T103000")
        )
    )["ppol_filtered"].mean(dim="time")
    - ppol_noise_profile
)
ref_profile = denoised.sel(range=slice(100, 200))
B, A = np.polyfit(ref_profile.range.values, np.log(ref_profile).values, deg=1)
df_fc["fc_ppol"] = np.exp(
    -(
        np.log(denoised.sel(range=slice(None, 200)))
        - (A + B * denoised.sel(range=slice(None, 200)).range)
    )
)

# xpol
df_fc_100["xpol_filtered"] = (
    ("time", "range"),
    denoise_fft(df_fc_100.x_pol, outlier_mask),
)
df_fc["xpol_filtered"] = xr.concat(
    [df_fc_100["xpol_filtered"], df_fc.where(noise.range > 100, drop=True)["p_pol"]],
    dim="range",
)

denoised = (
    df_fc.sel(
        time=slice(
            pd.to_datetime("2023-09-26T100000"), pd.to_datetime("2023-09-26T103000")
        )
    )["xpol_filtered"].mean(dim="time")
    - xpol_noise_profile
)
ref_profile = denoised.sel(range=slice(100, 200))
B, A = np.polyfit(ref_profile.range.values, np.log(ref_profile).values, deg=1)
df_fc["fc_xpol"] = np.exp(
    -(
        np.log(denoised.sel(range=slice(None, 200)))
        - (A + B * denoised.sel(range=slice(None, 200)).range)
    )
)

# %%
df_test =  xr.open_mfdataset(
    glob.glob("/media/viet/CL61/studycase/kenttarova/20231002/*.nc")
)

df_test_100 = df_test.sel(
    range=slice(None, 100))

# ppol
df_test_100["ppol_filtered"] = (
    ("time", "range"),
    denoise_fft(df_test_100.p_pol, outlier_mask),
)
df_test["ppol_filtered"] = xr.concat(
    [df_test_100["ppol_filtered"], df_test.where(noise.range > 100, drop=True)["p_pol"]],
    dim="range",
)

# xpol
df_test_100["xpol_filtered"] = (
    ("time", "range"),
    denoise_fft(df_test_100.x_pol, outlier_mask),
)
df_test["xpol_filtered"] = xr.concat(
    [
        df_test_100["xpol_filtered"],
        df_test.where(noise.range > 100, drop=True)["x_pol"],
    ],
    dim="range",
)

# %%
fig, ax = plt.subplots()
ax.pcolormesh(
    df_test.time,
    df_test.sel(range=slice(None, 200)).range,
    (
        (df_test.sel(range=slice(None, 200))["xpol_filtered"]
        * df_fc["fc_xpol"])
        / (df_test.sel(range=slice(None, 200))["ppol_filtered"]
        * df_fc["fc_ppol"])
    ).T,
    shading="auto",
    vmin=0,
    vmax=0.5,
)
ax.pcolormesh(
    df_test.time,
    df_test.sel(range=slice(200, None)).range,
    (
        df_test.sel(range=slice(200, None))["xpol_filtered"]
        / df_test.sel(range=slice(200, None))["ppol_filtered"]
    ).T,
    shading="auto",
    vmin=0,
    vmax=0.5,
)

# %%
fig, ax = plt.subplots()
ax.pcolormesh(
    df_test.time,
    df_test.range,
    (
        df_test["xpol_filtered"]
        / df_test["ppol_filtered"]
    ).T,
    shading="auto",
    vmin=0,
    vmax=0.5,
)


# %%
signal = df_test.p_pol
f_signal = fft.stft(signal.values, axis=0)
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
ppol_filtered = signal_filtered
df_test["ppol_filtered"] = (("time", "range"), ppol_filtered)
df_test["ppol_filtered"] = df_test.ppol_filtered - noise_profile

# %%
noise = df.sel(
    range=slice(None, 300),
    time=slice(
        pd.to_datetime("2023-09-26T111500"), pd.to_datetime("2023-09-26T161000")
    ),
).x_pol
# %%
fft = ShortTimeFFT(
    win=get_window("hamm", 256),
    hop=128,
    fs=10,
)

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
noise["noise_filtered"] = (("time", "range"), noise_filtered)
signal = df1.x_pol
f_signal = fft.stft(signal.values, axis=0)
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
xpol_filtered = signal_filtered
df1["xpol_filtered"] = (("time", "range"), xpol_filtered)

# %%
noise_profile = noise.sel(
    time=slice(pd.to_datetime("2023-09-26T120000"), pd.to_datetime("2023-09-26T150000"))
)["noise_filtered"].mean(dim="time")
denoised = (
    df1.sel(
        time=slice(
            pd.to_datetime("2023-09-26T100000"), pd.to_datetime("2023-09-26T103000")
        )
    )["xpol_filtered"].mean(dim="time")
    - noise_profile
)
# %%
ref_profile = denoised.sel(range=slice(100, 200))
B, A = np.polyfit(ref_profile.range.values, np.log(ref_profile).values, deg=1)
f_c_x = np.exp(
    -(
        np.log(denoised.sel(range=slice(None, 200)))
        - (A + B * denoised.sel(range=slice(None, 200)).range)
    )
)


# %%
signal = df_test.x_pol
f_signal = fft.stft(signal.values, axis=0)
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
xpol_filtered = signal_filtered
df_test["xpol_filtered"] = (("time", "range"), xpol_filtered)
df_test["xpol_filtered"] = df_test.xpol_filtered - noise_profile



# %%
fig, ax = plt.subplots()
ax.pcolormesh(
    df.time, df.range, (df.x_pol / df.p_pol).T, shading="auto", vmin=0, vmax=0.5
)
ax.set_ylim(0, 300)

# %%
(
    df_test.sel(range=slice(None, 200))["xpol_filtered"]
    * f_c_x
    / df_test.sel(range=slice(None, 200))["ppol_filtered"]
    * f_c_p
).isel(time=500).plot()
(
    df_test.sel(range=slice(None, 200))["x_pol"]
    / df_test.sel(range=slice(None, 200))["p_pol"]
).isel(time=500).plot()

# %%
df = xr.open_mfdataset(glob.glob("/media/viet/CL61/studycase/kenttarova/20231002/*.nc"))

signal = df.sel(range=slice(None, 300)).p_pol
f_signal = fft.stft(signal.values, axis=0)
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
ppol_filtered = signal_filtered
signal["ppol_filtered"] = (("time", "range"), ppol_filtered)
# signal["ppol_filtered"] = signal["ppol_filtered"] - noise_profile
# %%
fig, ax = plt.subplots(figsize=(8, 6), sharex=True, constrained_layout=True)
ax.pcolormesh(
    signal.time,
    signal.sel(range=slice(None, 200)).range,
    signal.sel(range=slice(None, 200)).ppol_filtered.T * f_c,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
    shading="auto",
)
ax.pcolormesh(
    signal.time,
    signal.sel(range=slice(200, None)).range,
    signal.sel(range=slice(200, None)).ppol_filtered.T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
    shading="auto",
)
# %%
fig, ax = plt.subplots(figsize=(8, 6), sharex=True, constrained_layout=True)
ax.pcolormesh(
    signal.time,
    signal.sel(range=slice(None, 200)).range,
    signal.sel(range=slice(None, 200)).T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
    shading="auto",
)
ax.pcolormesh(
    signal.time,
    signal.sel(range=slice(200, None)).range,
    signal.sel(range=slice(200, None)).T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
    shading="auto",
)
# %%
