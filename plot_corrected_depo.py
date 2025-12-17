import numpy as np
import pandas as pd
import glob
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import string
from cl61.func.study_case import process_raw, background_noise
from matplotlib.colors import LogNorm
import matplotlib.ticker as mtick
from cl61.func import rayleigh

myFmt = mdates.DateFormatter("%Y\n%m-%d\n%H:%M")

# %%
date = "20240604"
time_slice = slice("2024-06-04T13:00", "2024-06-04T17:00")
file_dir = f"/media/viet/CL61/studycase/kenttarova/{date}/"
df_sample = process_raw(file_dir, f"{date} 130000", f"{date} 170000")
bk_noise = background_noise("kenttarova", date)  # same time frame
bk_noise = (
    bk_noise.set_index("datetime")
    .between_time("13:00", "17:00")
    .reset_index()
    .mean(numeric_only=True)
)
# %%
ref_mean = xr.open_dataset(
    "/media/viet/CL61/calibration/result/kenttarova/calibration_mean.nc"
)
df_mean_ref_sample = ref_mean.interp(
    internal_temperature_bins=df_sample.internal_temperature_bins,
    method="linear",
    kwargs={"fill_value": "extrapolate", "bounds_error": False},
).drop_vars("internal_temperature_bins")
ref_std = xr.open_dataset(
    "/media/viet/CL61/calibration/result/kenttarova/calibration_std.nc"
)
df_std_ref_sample = ref_std.interp(
    internal_temperature_bins=df_sample.internal_temperature_bins,
    method="linear",
    kwargs={"fill_value": "extrapolate", "bounds_error": False},
).drop_vars("internal_temperature_bins")

# %%
df_sample = df_sample.sel(range=slice(50, 15000))
df_sample["ppol_c"] = df_sample["ppol_r"] - df_mean_ref_sample["ppol_ref"]
df_sample["xpol_c"] = df_sample["xpol_r"] - df_mean_ref_sample["xpol_ref"]

df_sample["depo_c"] = df_sample["xpol_c"] / df_sample["ppol_c"]
df_sample["beta_c"] = (df_sample["ppol_c"] + df_sample["xpol_c"]) * (df_sample.range**2)
df_sample["beta_0"] = (df_sample["ppol_r"] + df_sample["xpol_r"]) * (df_sample.range**2)
df_sample["depo_0"] = df_sample["xpol_r"] / df_sample["ppol_r"]
# For depo, no need range correction, as result for uncertainty will be the same
instrument_1012 = (
    df_std_ref_sample["xpol_r"]
    .sel(range=slice(10000, 12000))
    .mean(dim="range", skipna=True)
)
solar_1012 = xr.where(
    instrument_1012 < bk_noise["cross_std"],
    np.sqrt(bk_noise["cross_std"] ** 2 - instrument_1012**2),
    0,
)
df_sample["xpol_std"] = np.sqrt(df_std_ref_sample["xpol_r"] ** 2 + solar_1012**2)

instrument_1012 = (
    df_std_ref_sample["ppol_r"]
    .sel(range=slice(10000, 12000))
    .mean(dim="range", skipna=True)
)

solar_1012 = xr.where(
    instrument_1012 < bk_noise["co_std"],
    np.sqrt(bk_noise["co_std"] ** 2 - instrument_1012**2),
    0,
)
df_sample["ppol_std"] = np.sqrt(df_std_ref_sample["ppol_r"] ** 2 + solar_1012**2)

df_sample["beta_v_std"] = (
    np.sqrt(df_sample["ppol_std"] ** 2 + df_sample["xpol_std"] ** 2)
) * df_sample.range**2  #  range correction, think carefully here

df_sample["depo_c_std"] = np.abs(df_sample["depo_c"]) * np.sqrt(
    (df_sample["xpol_std"] / df_sample["xpol_c"]) ** 2
    + (df_sample["ppol_std"] / df_sample["ppol_c"]) ** 2
)
# %%
model = xr.open_dataset(
    glob.glob(f"/media/viet/CL61/studycase/kenttarova/{date}/weather/*ecmwf.nc")[0]
)
model = model.sel(time=time_slice)
model = model[["temperature", "pressure", "q", "height"]]
model = model.interp(time=df_sample.time, method="nearest")


# %% inversion for aerosol only
def interp_to_height(x, z, z_new):
    return np.interp(z_new, z, x)


z_new = xr.DataArray(df_sample.range.values, dims=["range"])
for x in ["q", "temperature", "pressure"]:
    model[x] = xr.apply_ufunc(
        interp_to_height,
        model[x],  # (time, level)
        model.height,  # (time, level)
        z_new,  # (range)
        input_core_dims=[["level"], ["level"], ["range"]],
        output_core_dims=[["range"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[model[x].dtype],
    )
    model[x] = model[x].assign_coords(range=z_new)

mol_scatter = rayleigh.molecular_backscatter(
    np.pi,
    model["temperature"],
    model["pressure"] / 100,  # Pa to hPa
)
beta_mol = mol_scatter / 1000

depo_mol = rayleigh.depo(
    rayleigh.f(0.91055, 425, rayleigh.humidity_conversion(model["q"]))
)
df_sample["beta_p"] = rayleigh.forward(
    df_sample.beta_c,
    beta_mol,
    50,
    1 / 1,
    df_sample.range,
)
df_sample["beta_p_std"] = rayleigh.forward_sigma(
    df_sample["beta_p"],
    beta_mol,
    50,
    1 / 1,
    df_sample.range,
    df_sample["beta_v_std"],
)

# %%
depo_volume = df_sample["xpol_c"] / df_sample["ppol_c"]
beta_ratio = rayleigh.backscatter_ratio(df_sample["beta_p"], beta_mol)
df_sample["depo_aerosol"] = rayleigh.depo_aerosol(depo_volume, depo_mol, beta_ratio)
df_sample["depo_aerosol_sigma"] = rayleigh.depo_aersosol_sigma(
    depo_volume, depo_mol, beta_ratio, df_sample["depo_c_std"], df_sample["beta_p_std"]
)

# %%
fig, ax = plt.subplots(
    4, 2, figsize=(12, 9), constrained_layout=True, sharey="row", sharex=True
)
# beta corrected
p = ax[0, 0].pcolormesh(
    df_sample.time,
    df_sample.range,
    df_sample["beta_c"].T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
fig.colorbar(p, ax=ax[0, 0], label=r"$\beta'_v$")
p = ax[0, 1].pcolormesh(
    df_sample.time,
    df_sample.range,
    (df_sample["beta_v_std"] ** 2).T,
    norm=LogNorm(vmin=1e-16, vmax=1e-12),
)
fig.colorbar(p, ax=ax[0, 1], label=r"$\sigma^2_{\beta'_v}$")
ax[0, 1].set_ylim(0, 4000)

# depo corrected
p = ax[1, 0].pcolormesh(
    df_sample.time, df_sample.range, df_sample["depo_c"].T, vmin=0, vmax=0.3
)
fig.colorbar(p, ax=ax[1, 0], label=r"$\delta_{v}$")
p = ax[1, 1].pcolormesh(
    df_sample.time, df_sample.range, (df_sample["depo_c_std"] ** 2).T, vmin=0, vmax=0.1
)
fig.colorbar(p, ax=ax[1, 1], label=r"$\sigma^2_{\delta_{v}}$")
ax[1, 1].set_ylim(0, 4000)

# beta aerosol
p = ax[2, 0].pcolormesh(
    df_sample.time,
    df_sample.range,
    df_sample["beta_p"].T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
fig.colorbar(p, ax=ax[2, 0], label=r"$\beta_p$")
p = ax[2, 1].pcolormesh(
    df_sample.time,
    df_sample.range,
    (df_sample["beta_p_std"] ** 2).T,
    norm=LogNorm(vmin=1e-16, vmax=1e-12),
)
fig.colorbar(p, ax=ax[2, 1], label=r"$\sigma^2_{\beta_p}$")
ax[2, 1].set_ylim(0, 1000)

# depo aerosol
p = ax[3, 0].pcolormesh(
    df_sample.time, df_sample.range, df_sample["depo_aerosol"].T, vmin=0, vmax=0.3
)
fig.colorbar(p, ax=ax[3, 0], label=r"$\delta_{p}$")
p = ax[3, 1].pcolormesh(
    df_sample.time,
    df_sample.range,
    (df_sample["depo_aerosol_sigma"] ** 2).T,
    vmin=0,
    vmax=0.1,
)
fig.colorbar(p, ax=ax[3, 1], label=r"$\sigma^2_{\delta_{p}}$")
ax[3, 1].set_ylim(0, 1000)
ax[3, 0].xaxis.set_major_formatter(myFmt)
ax[3, 0].xaxis.set_major_locator(mdates.HourLocator(interval=1))
for ax_ in ax[:, 0]:
    ax_.set_ylabel("Range (m)")

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.03,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
fig.savefig(
    "/media/viet/CL61/img/studycase_depo_corrected.png", bbox_inches="tight", dpi=600
)

# %%
df_plot = df_sample.sel(time="2024-06-04T13:00", method="nearest")
fig, ax = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True, sharey=True)
ax[0, 0].plot(df_plot["beta_0"], df_plot.range, ".", label=r"$\beta$")
ax[0, 0].plot(df_plot["beta_c"], df_plot.range, ".", label=r"$\beta_v$")
ax[0, 0].plot(df_plot["beta_p"], df_plot.range, ".", label=r"$\beta_p$")
# ax[0, 0].set_xscale('log')
ax[0, 0].set_xlim(1e-8, 4e-7)
ax[0, 0].set_xlabel(r"$\beta$")

ax[0, 1].plot(
    df_plot["beta_v_std"] ** 2, df_plot.range, ".", label=r"$\sigma^2_{\beta_v}$"
)
ax[0, 1].plot(
    df_plot["beta_p_std"] ** 2, df_plot.range, ".", label=r"$\sigma^2_{\beta_p}$"
)
# ax[0, 1].set_xscale("log")
ax[0, 1].set_xlim(1e-17, 4e-14)
ax[0, 1].set_xlabel(r"$\sigma^2_{\beta}$")

ax[1, 0].plot(df_plot["depo_0"], df_plot.range, ".", label=r"$\delta$")
ax[1, 0].plot(df_plot["depo_c"], df_plot.range, ".", label=r"$\delta_v$")
ax[1, 0].plot(df_plot["depo_aerosol"], df_plot.range, ".", label=r"$\delta_p$")
ax[1, 0].set_xlim(0, 0.3)
ax[1, 0].set_xlabel(r"$\delta$")


ax[1, 1].plot(
    df_plot["depo_c_std"] ** 2, df_plot.range, ".", label=r"$\sigma^2_{\delta_v}$"
)
ax[1, 1].plot(
    df_plot["depo_aerosol_sigma"] ** 2,
    df_plot.range,
    ".",
    label=r"$\sigma^2_{\delta_p}$",
)
ax[1, 1].set_xlim(0, 0.1)
ax[1, 1].set_xlabel(r"$\sigma^2_{\delta}$")

for ax_ in ax[:, 0]:
    ax_.set_ylabel("Range (m)")

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.03,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
    ax_.legend()
    ax_.set_ylim(50, 1000)
    ax_.grid()
fig.savefig(
    "/media/viet/CL61/img/studycase_depo_corrected_profile.png",
    bbox_inches="tight",
    dpi=600,
)
# %%
df_plot = df_sample.sel(time="2024-06-04T13:00", method="nearest")
fig, ax = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True, sharey=True)
ax[0].plot(
    df_plot["beta_0"] - df_plot["beta_c"], df_plot.range, ".", label=r"$\beta$"
)
ax[1].plot(df_plot["depo_0"] - df_plot["depo_c"], df_plot.range, ".", label=r"$\delta$")
for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.03,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
    ax_.legend()
    ax_.set_ylim(50, 1000)
    ax_.grid()

ax[1].set_xlim(0, 0.005)
ax[0].set_xlim(0, 4e-9)
# %%
