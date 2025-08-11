import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
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
ppol_noise_filtered, ppol_outlier_mask = analyze_noise(noise_100.p_pol)
noise_100["ppol_filtered"] = (("time", "range"), ppol_noise_filtered)
noise["ppol_filtered"] = xr.concat(
    [noise_100["ppol_filtered"], noise.where(noise.range > 100, drop=True)["p_pol"]],
    dim="range",
)
ppol_noise_profile = noise.sel(
    time=slice(pd.to_datetime("2023-09-26T120000"), pd.to_datetime("2023-09-26T150000"))
)["ppol_filtered"].mean(dim="time")

# xpol
xpol_noise_filtered, xpol_outlier_mask = analyze_noise(noise_100.x_pol)
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
    denoise_fft(df_fc_100.p_pol, ppol_outlier_mask),
)
df_fc["ppol_filtered"] = xr.concat(
    [df_fc_100["ppol_filtered"], df_fc.where(df_fc.range > 100, drop=True)["p_pol"]],
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
    denoise_fft(df_fc_100.x_pol, xpol_outlier_mask),
)
df_fc["xpol_filtered"] = xr.concat(
    [df_fc_100["xpol_filtered"], df_fc.where(df_fc.range > 100, drop=True)["x_pol"]],
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
df_test = xr.open_mfdataset(
    glob.glob("/media/viet/CL61/studycase/kenttarova/20231002/*.nc")
)
# df_test = df
df_test_100 = df_test.sel(range=slice(None, 100))

# ppol
df_test_100["ppol_filtered"] = (
    ("time", "range"),
    denoise_fft(df_test_100.p_pol, ppol_outlier_mask),
)
df_test["ppol_filtered"] = xr.concat(
    [
        df_test_100["ppol_filtered"],
        df_test.where(df_test.range > 100, drop=True)["p_pol"],
    ],
    dim="range",
)

# xpol
df_test_100["xpol_filtered"] = (
    ("time", "range"),
    denoise_fft(df_test_100.x_pol, xpol_outlier_mask),
)
df_test["xpol_filtered"] = xr.concat(
    [
        df_test_100["xpol_filtered"],
        df_test.where(df_test.range > 100, drop=True)["x_pol"],
    ],
    dim="range",
)

# %% Plot depo corrected
fig, ax = plt.subplots(
    2, 1, figsize=(12, 6), sharex=True, constrained_layout=True, sharey=True
)
ax[0].pcolormesh(
    df_test.time,
    df_test.range,
    (df_test["x_pol"] / df_test["p_pol"]).T,
    shading="auto",
    vmin=0,
    vmax=0.5,
)

ax[1].pcolormesh(
    df_test.time,
    df_test.sel(range=slice(None, 200)).range,
    (
        (df_test.sel(range=slice(None, 200))["xpol_filtered"] * df_fc["fc_xpol"])
        / (df_test.sel(range=slice(None, 200))["ppol_filtered"] * df_fc["fc_ppol"])
    ).T,
    shading="auto",
    vmin=0,
    vmax=0.5,
)
ax[1].pcolormesh(
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
ax[0].set_ylim(0, 500)

# %%
fig, ax = plt.subplots(
    2, 1, figsize=(12, 6), sharex=True, constrained_layout=True, sharey=True
)
ax[0].pcolormesh(
    df_test.sel(range=slice(None, 200)).time,
    df_test.sel(range=slice(None, 200)).range,
    (df_test.sel(range=slice(None, 200))["ppol_filtered"] - ppol_noise_profile).T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
    shading="auto",
)

ax[1].pcolormesh(
    df_test.sel(range=slice(None, 200)).time,
    df_test.sel(range=slice(None, 200)).range,
    (df_test.sel(range=slice(None, 200))["xpol_filtered"] - xpol_noise_profile).T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
    shading="auto",
)

ax[0].set_ylim(0, 200)

# %%
fig, ax = plt.subplots(figsize=(12, 6), sharex=True, constrained_layout=True)
ax.pcolormesh(
    df_test.time,
    df_test.sel(range=slice(None, 200)).range,
    (
        (
            (df_test.sel(range=slice(None, 200))["xpol_filtered"] - xpol_noise_profile)
            * df_fc["fc_xpol"]
        )
        / (
            (df_test.sel(range=slice(None, 200))["ppol_filtered"] - ppol_noise_profile)
            * df_fc["fc_ppol"]
        )
    ).T,
    shading="auto",
    vmin=0,
    vmax=0.5,
)
# %%
depo_new = (
    (df_test.sel(range=slice(None, 200))["xpol_filtered"] - xpol_noise_profile)
    * df_fc["fc_xpol"]
) / (
    (df_test.sel(range=slice(None, 200))["ppol_filtered"] - ppol_noise_profile)
    * df_fc["fc_ppol"]
)
depo = (
    df_test.sel(range=slice(None, 200))["xpol_filtered"]
    / df_test.sel(range=slice(None, 200))["ppol_filtered"]
)
# %%
fig, ax = plt.subplots()
ax.plot(depo_new.isel(time=5), depo_new.range)
ax.set_xlim(0, 0.1)

# %%
fig, ax = plt.subplots()
ax.plot(df_test.time, df_test.x_pol.isel(range=5), label="xpol")
ax.plot(df_test.time, df_test.xpol_filtered.isel(range=5), label="xpol_filtered")
ax.legend()
ax.set_yscale("log")
ax.set_ylim(1e-8, 1e-6)

# %%
fig, ax = plt.subplots()
ax.plot(depo_new.time, depo_new.isel(range=5))
ax.set_ylim(0, 0.1)
# %%
fig, ax = plt.subplots()
ax.plot(df_test.time, df_test.xpol_filtered.isel(range=5), label="xpol_filtered")
ax.set_ylim(-1e-7, 1e-7)
# %%
fig, ax = plt.subplots()
ax.plot(
    depo.sel(time=pd.to_datetime("2023-10-02T210000"), method="nearest"), depo.range
)
ax.plot(
    depo_new.sel(time=pd.to_datetime("2023-10-02T210000"), method="nearest"),
    depo_new.range,
)
# %%
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_test.time, df_test.x_pol.isel(range=5), label="x_pol")
ax.set_ylim(-1e-6, 1e-6)
ax.grid()
# %%
