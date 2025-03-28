import numpy as np
import pandas as pd
import glob
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cl61.func.calibration_T import temperature_ref
import string
from cl61.func.study_case import process_raw, background_noise

myFmt = mdates.DateFormatter("%Y\n%m-%d\n%H:%M")

# %%
file_dir = "/media/viet/CL61/studycase/kenttarova/20231009/"
df_sample = process_raw(file_dir, "20231009 210000", "20231009 230000")
bk_noise = background_noise("kenttarova", "20231009")
bk_noise = (
    bk_noise.set_index("datetime")
    .between_time("21:00", "23:00")
    .reset_index()
    .mean(numeric_only=True)
)
# %%
files = glob.glob("/media/viet/CL61/calibration/kenttarova/merged/*.nc")
df = xr.open_mfdataset(files)
df_mean, df_std = temperature_ref(df)

df_mean_ref_sample = df_mean.sel(
    internal_temperature_bins=df_sample.internal_temperature_bins
).drop_vars("internal_temperature_bins")
df_std_ref_sample = df_std.sel(
    internal_temperature_bins=df_sample.internal_temperature_bins
).drop_vars("internal_temperature_bins")


# bk_ppol_mean = (
#     bk_noise["co_mean"]
#     - df_mean_ref_sample["ppol_r"]
#     .sel(range=slice(8000, 10000))
#     .mean(dim="range")
# )
# bk_xpol_mean = (
#     bk_noise["cross_mean"]
#     - df_mean_ref_sample["xpol_r"]
#     .sel(range=slice(8000, 10000))
#     .mean(dim="range")
# )
# bk_ppol_std = np.sqrt(
#     bk_noise["co_std"] ** 2
#     - (df_std_ref_sample["ppol_r"].sel(range=slice(8000, 10000)).mean(dim="range", skipna=True))
#     ** 2
# )

# bk_xpol_std = np.sqrt(
#     bk_noise["cross_std"] ** 2
#     - (df_std_ref_sample["xpol_r"].sel(range=slice(8000, 10000)).mean(dim="range", skipna=True))
#     ** 2
# )

df_sample["ppol_c"] = df_sample["ppol_r"] - df_mean_ref_sample["ppol_ref"]
df_sample["xpol_c"] = df_sample["xpol_r"] - df_mean_ref_sample["xpol_ref"]

df_sample["depo_c"] = df_sample["xpol_c"] / df_sample["ppol_c"]

# df_sample["xpol_std"] = np.sqrt(df_std_ref_sample["xpol_r"]**2 + bk_xpol_std**2)
# df_sample["ppol_std"] = np.sqrt(df_std_ref_sample["ppol_r"]**2 + bk_ppol_std**2)

df_sample["xpol_std"] = df_std_ref_sample["xpol_r"]
df_sample["ppol_std"] = df_std_ref_sample["ppol_r"]

df_sample["depo_c_std"] = np.abs(df_sample["depo_c"]) * np.sqrt(
    (df_sample["xpol_std"] / df_sample["xpol_c"]) ** 2
    + (df_sample["ppol_std"] / df_sample["ppol_c"]) ** 2
)

# %%
fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True, constrained_layout=True)

ax[0].plot(df_sample["ppol_r"].mean(dim="time"), df_sample["range"], label="original")
ax[0].plot(df_sample["ppol_c"].mean(dim="time"), df_sample["range"], label="corrected")
ax[0].set_xlim([-1e-10, 5e-10])

ax[1].plot(df_sample["xpol_r"].mean(dim="time"), df_sample["range"], label="original")
ax[1].plot(df_sample["xpol_c"].mean(dim="time"), df_sample["range"], label="corrected")
ax[1].set_xlim([-1e-13, 5e-13])

ax[2].plot(
    df_sample["xpol_r"].mean(dim="time") / df_sample["ppol_r"].mean(dim="time"),
    df_sample["range"],
    ".",
    label="original",
)
ax[2].errorbar(
    x=df_sample["xpol_c"].mean(dim="time") / df_sample["ppol_c"].mean(dim="time"),
    y=df_sample.range,
    xerr=np.sqrt((df_sample["depo_c_std"] ** 2).sum(dim="time")) / df_sample.time.size,
    fmt=".",
    label="corrected",
)

ax[2].set_xlim([-0.01, 0.008])

for n, ax_ in enumerate(ax.flatten()):
    ax_.grid()
    ax_.legend()
    ax_.text(
        -0.0,
        1.03,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
ax[0].set_ylim([0, 1200])
ax[0].set_ylabel("Range (m)")
ax[0].set_xlabel("ppol/r²")
ax[1].set_xlabel("xpol/r²")
ax[2].set_xlabel(r"$\delta$")
fig.savefig(
    "/media/viet/CL61/img/studycase_depo_profile.png", bbox_inches="tight", dpi=600
)

# %%
mask = df_sample["depo_c_std"] < 0.01
fig, ax = plt.subplots(
    1, 3, figsize=(10, 3), constrained_layout=True, sharex=True, sharey=True
)
p = ax[0].pcolormesh(
    df_sample["time"],
    df_sample["range"],
    (df_sample["x_pol"] / df_sample["p_pol"]).T,
    vmin=0,
    vmax=0.004,
)
cbar = fig.colorbar(p, ax=ax[0])
cbar.set_label(r"$\delta_{original}$")

p = ax[1].pcolormesh(
    df_sample["time"],
    df_sample["range"],
    df_sample["depo_c"].where(mask).T,
    vmin=0,
    vmax=0.004,
)
cbar = fig.colorbar(p, ax=ax[1])
cbar.set_label(r"$\delta_{corrected}$")

p = ax[2].pcolormesh(
    df_sample["time"],
    df_sample["range"],
    df_sample["depo_c_std"].where(mask).T,
    vmin=0,
    vmax=0.01,
)
cbar = fig.colorbar(p, ax=ax[2])
cbar.set_label(r"$\sigma_{\delta}$")
for ax_ in ax.flatten():
    ax_.xaxis.set_major_formatter(myFmt)
    ax_.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax[0].set_ylim([0, 1200])

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.03,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
ax[0].set_ylabel("Range (m)")
fig.savefig(
    "/media/viet/CL61/img/studycase_depo_corrected.png", bbox_inches="tight", dpi=600
)
