import numpy as np
import glob
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import string
from cl61.func.study_case import process_raw, background_noise
from cl61.func import rayleigh

myFmt = mdates.DateFormatter("%Y\n%m-%d\n%H:%M")

# %%
date = "20231002"
time_slice = slice("2023-10-02T11:00", "2023-10-02T15:00")
file_dir = f"/media/viet/CL61/studycase/kenttarova/{date}/"
df_sample = process_raw(file_dir, f"{date} 110000", f"{date} 150000")

# %%
bk_noise = background_noise("kenttarova", date)  # same time frame
# bk_noise = (
#     bk_noise.set_index("datetime")
#     .between_time("13:00", "17:00")
#     .reset_index()
#     .mean(numeric_only=True)
# )
bk_noise = bk_noise.rename(columns={"datetime": "time"})
bk_noise = bk_noise.set_index("time").to_xarray()
bk_noise = bk_noise.interp(time=df_sample.time, method="linear")

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

df_sample["beta_v_std"] = np.sqrt(
    (df_sample["ppol_std"] ** 2 + df_sample["xpol_std"] ** 2)
    * df_sample.range**4  #  range correction, think carefully here
    + 0.01 * df_sample["beta_c"] ** 2
)

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
    depo_volume,
    depo_mol,
    beta_ratio,
    df_sample["depo_c_std"],
    df_sample["beta_p_std"],
    beta_mol,
)

# %%
df_plot = df_sample.sel(time=slice("2023-10-02T12:30", "2023-10-02T13:30")).mean(
    dim="time"
)
fig, ax = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True, sharey=True)
ax[0].plot(
    df_plot["xpol_r"] * df_plot.range**2,
    df_plot.range,
    label=r"$^\perp \beta'$",
)
ax[0].plot(
    df_plot["xpol_c"] * df_plot.range**2,
    df_plot.range,
    label=r"$^\perp \beta'_\mathrm{corrected}$",
)
ax[0].set_xlim(-1e-9, 5e-9)
ax[0].set_xlabel(r"$^\perp \beta'$ [a.u.]")
# ax[0].set_xlim(-1e-13, 5e-13)

ax[1].plot(
    df_plot["ppol_r"] * df_plot.range**2,
    df_plot.range,
    label=r"$^\parallel \beta'$",
)
ax[1].plot(
    df_plot["ppol_c"] * df_plot.range**2,
    df_plot.range,
    label=r"$^\parallel \beta'_\mathrm{corrected}$",
)
ax[1].set_xlim(-1e-7, 1e-6)
ax[1].set_xlabel(r"$^\parallel \beta'$ [a.u.]")

ax[2].plot(
    df_plot["depo_0"],
    df_plot.range,
    label=r"$\delta$",
)
ax[2].plot(
    df_plot["depo_c"],
    df_plot.range,
    label=r"$\delta_\mathrm{corrected}$",
)
ax[2].set_xlim(-0.001, 0.01)
ax[2].set_xlabel(r"$\delta$")
ax[0].set_ylabel("Range [m]")

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.03,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
    ax_.legend()
    ax_.set_ylim(0, 500)
    ax_.grid()
fig.savefig(
    "/media/viet/CL61/img/corrected_instrumental_profiles.png",
    dpi=600,
    bbox_inches=None,
)
# %%
