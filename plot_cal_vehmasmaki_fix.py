import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from cl61.func.calibration_T import temperature_ref
from cl61.func.study_case import process_raw
from matplotlib.colors import LogNorm
import string
# %%

file_dir = "/media/viet/CL61/studycase/vehmasmaki/20241209/"
df_sample = process_raw(file_dir, "20241209 000000", "20241209 120000")

df = xr.open_dataset("/media/viet/CL61/calibration/vehmasmaki/merged/20241127.nc")
df_mean, _ = temperature_ref(df)

full_T = dict(internal_temperature_bins=np.arange(-10, 40))
df_mean_full, _ = xr.align(
    df_mean, xr.DataArray(dims=full_T, coords=full_T), join="outer"
)

df_mean_ref_sample = df_mean_full.sel(
    internal_temperature_bins=df_sample.internal_temperature_bins
).drop_vars("internal_temperature_bins")
# df_std_ref_sample = df_std.sel(
#     internal_temperature_bins=df_sample.internal_temperature_bins
# ).drop_vars("internal_temperature_bins")

# %%
df_sample["ppol_c"] = (
    df_sample["p_pol"]
    - df_mean_ref_sample["ppol_ref"] * df_mean_ref_sample["range"] ** 2
)
df_sample["xpol_c"] = (
    df_sample["x_pol"]
    - df_mean_ref_sample["xpol_ref"] * df_mean_ref_sample["range"] ** 2
)

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True, constrained_layout=True)
ax[0].pcolormesh(
    df_sample.time,
    df_sample.range,
    df_sample["p_pol"].T,
    shading="nearest",
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
ax[1].pcolormesh(
    df_sample.time,
    df_sample.range,
    df_sample["ppol_c"].T,
    shading="nearest",
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
fig.savefig(
    "/media/viet/CL61/img/studycase_vehmasmaki_20241209.png",
    dpi=600,
    bbox_inches="tight",
)
# %%
profile = df_sample.isel(time=2)
ref_profile = df_mean.sel(internal_temperature_bins=profile.internal_temperature_bins)

fig, ax = plt.subplots(1, 3, sharey=True, figsize=(9, 3), sharex=True)
ax[0].plot(profile.ppol_r, df_sample.range, ".")
ax[1].plot(profile.ppol_r - ref_profile.ppol_ref, df_sample.range, ".")
ax[2].plot(ref_profile.ppol_ref, df_sample.range, ".")
ax[0].set_xlim([-2e-13, 5e-14])

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.03,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
    ax_.grid()
fig.savefig(
    "/media/viet/CL61/img/studycase_vehmasmaki_20241209.png",
    dpi=600,
    bbox_inches="tight",
)
# %%
