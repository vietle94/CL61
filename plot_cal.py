import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib as mpl
from cl61.func.noise import noise_filter

# %%
site = "Kenttarova"
file_dir = f"/media/viet/CL61/Calibration/{site}/Summary/Data_merged/"
df_signal = xr.open_mfdataset(glob.glob(file_dir + "*_signal.nc"))
df_diag = xr.open_mfdataset(glob.glob(file_dir + "*_diag.nc"))

df_diag = df_diag.reindex(time=df_signal.time.values, method="nearest", tolerance="8s")
df_diag = df_diag.dropna(dim="time")

df = df_diag.merge(df_signal, join="inner")

df["ppol_r"] = df.p_pol / (df.range**2)
df["xpol_r"] = df.x_pol / (df.range**2)
temp_range = np.arange(
    np.floor(df.internal_temperature.min(skipna=True).values),
    np.ceil(df.internal_temperature.max(skipna=True)).values + 1,
)
df["ppol_r"] = df.p_pol / (df.range**2)
df["xpol_r"] = df.x_pol / (df.range**2)

df_gr = df.groupby_bins("internal_temperature", temp_range)
df_mean = df_gr.mean(dim="time", skipna=True)
df_std = df_gr.std(dim="time", skipna=True)
df_count = df_gr.count(dim="time")
df_mean = df_mean.dropna(dim="internal_temperature_bins")
df_std = df_std.dropna(dim="internal_temperature_bins")
df_std.sel(range=20, method="nearest").ppol_r.values


my_c = mpl.colormaps["RdBu_r"](
    np.linspace(0, 1, df_mean.internal_temperature_bins.size)
)

fig, ax = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
for t, myc in zip(df_mean.internal_temperature_bins, my_c):
    t_mean = df_mean.sel(internal_temperature_bins=t)
    ppol_r = noise_filter(t_mean["ppol_r"])
    xpol_r = noise_filter(t_mean["xpol_r"])

    t_std = df_std.sel(internal_temperature_bins=t)
    ax[0, 0].plot(ppol_r, t_mean.range, color=myc, label=t.values)
    ax[0, 0].set_xlim(-1e-14, 1e-13)
    ax[0, 0].set_xlabel(r"$ppol/range^2$")

    ax[1, 0].plot(ppol_r, t_mean.range, color=myc, label=t.values)
    ax[1, 0].set_xlim(-1e-14, 1.5e-13)
    ax[1, 0].set_xlabel(r"$ppol/range^2$")
    ax[1, 0].set_ylim([0, 500])

    ax[0, 1].plot(xpol_r, t_mean.range, color=myc, label=t.values)
    ax[0, 1].set_xlim(-1e-14, 1e-13)
    ax[0, 1].set_xlabel(r"$xpol/range^2$")

    ax[1, 1].plot(xpol_r, t_mean.range, color=myc, label=t.values)
    ax[1, 1].set_xlim(-1e-14, 1.5e-13)
    ax[1, 1].set_xlabel(r"$xpol/range^2$")
    ax[1, 1].set_ylim([0, 500])

for ax_ in ax.flatten():
    ax_.legend()
    ax_.grid()
fig.savefig(
    f"/media/viet/CL61/Calibration/{site}/Summary/" + "internalT.png",
    bbox_inches="tight",
    dpi=600,
)

# %%
