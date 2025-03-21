import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import string
from cl61.func.calibration_T import temperature_ref, noise_filter
from cl61.func.study_case import process_raw, background_noise

# %%

file_dir = "/media/viet/CL61/studycase/vehmasmaki/20241209/"
df_sample = process_raw(file_dir, "20241209 000000", "20241209 120000")

df = xr.open_dataset("/media/viet/CL61/calibration/vehmasmaki/merged/20241127.nc")
df_mean, df_std = temperature_ref(df)

df_mean_ref_sample = df_mean.sel(
    internal_temperature_bins=df_sample.internal_temperature_bins
).drop_vars("internal_temperature_bins")
# df_std_ref_sample = df_std.sel(
#     internal_temperature_bins=df_sample.internal_temperature_bins
# ).drop_vars("internal_temperature_bins")

# %%
df_sample["ppol_c"] = df_sample["ppol_r"] - df_mean_ref_sample["ppol_ref"]
df_sample["xpol_c"] = df_sample["xpol_r"] - df_mean_ref_sample["xpol_ref"]

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True, constrained_layout=True)
ax[0].plot(df_sample["p_pol"], df_sample.range, ".")
ax[1].plot(df_sample["ppol_c"] * df["range"] ** 2, df_sample.range, ".")
