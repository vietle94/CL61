import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from cl61.func.calibration_T import temperature_ref, noise_filter, noise_filter_std
import glob
import string
from cl61.func.utils import polly_fernald, molecular_backscatter
import pandas as pd

# %%
radiosonde = pd.read_csv("/media/viet/CL61/2024030512-02836.csv")
radiosonde["range"] = radiosonde["geopotential height_m"] - 180
radiosonde["molBsc"] = molecular_backscatter(
    radiosonde["temperature_C"] + 273.15, radiosonde["pressure_hPa"]
)
radiosonde = radiosonde[radiosonde["range"] < 15720]
# %%
site = "kenttarova"
files = glob.glob(f"/media/viet/CL61/calibration/{site}/merged/*.nc")
df = xr.open_dataset(files[2])
df_mean, df_std, _ = temperature_ref(df)

# %%
file_dir = "/media/viet/CL61/studycase/kenttarova/20240305/"
df_case_full = xr.open_mfdataset(file_dir + "*.nc")
df_case_full = df_case_full.isel(range=slice(1, None))
df_case_full = df_case_full.sel(time=slice("2024-03-05T11:00", "2024-03-05T16:00"))
df_case = df_case_full.sel(time=slice("2024-03-05T11:00", "2024-03-05T12:00"))
df_case["ppol_r"] = df_case.p_pol / (df_case.range**2)

df_case_mean = df_case.mean(dim="time", skipna=True)
df_case_std = df_case.std(dim="time", skipna=True)
# profile = df_case.sel(time="2023-09-26T<10:00", method="nearest")

# %%
fig = plt.figure(layout="constrained", figsize=(8, 8))
ax_dict = fig.subplot_mosaic(
    [["time", "time"], ["mean_case", "mean_cal"], ["std_case", "std_cal"]],
    sharey=True,
)
p = ax_dict["time"].pcolormesh(
    df_case_full["time"],
    df_case_full["range"],
    df_case_full["p_pol"].T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
fig.colorbar(p, ax=ax_dict["time"], label=r"$\beta$ [m-1 sr-1]")
ax_dict["time"].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax_dict["time"].xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax_dict["time"].set_ylabel("Range [m]")

ax_dict["mean_case"].scatter(
    df_case_mean.ppol_r, df_case_mean.range, s=1, label=r"$\mu_{ppol/r²}$"
)
ax_dict["mean_case"].scatter(
    radiosonde["molBsc"] / 1000 / radiosonde["range"] ** 2,
    radiosonde["range"],
    s=1,
    label=r"$\beta_{mol}/r²$",
    c="C3",
)
ax_dict["mean_case"].set_ylabel("Range [m]")
ax_dict["mean_case"].legend(loc="upper right")

ax_dict["mean_cal"].scatter(
    df_mean.sel(internal_temperature_bins=18).ppol_r, df_mean.range, s=1, c="C0"
)

ax_dict["mean_cal"].scatter(
    df_mean.sel(internal_temperature_bins=18, range=slice(250, None)).ppol_ref,
    df_mean.sel(internal_temperature_bins=18, range=slice(250, None)).range,
    s=1,
    c="C1",
)

ax_dict["mean_cal"].scatter(
    df_mean.sel(internal_temperature_bins=18, range=slice(None, 250)).ppol_r,
    df_mean.sel(internal_temperature_bins=18, range=slice(None, 250)).range,
    s=1,
    c="C1",
)

ax_dict["mean_cal"].set_xlabel(r"$\mu_{ppol/r²}$")


ax_dict["std_case"].scatter(df_case_std.ppol_r, df_case_std.range, s=1)
ax_dict["std_case"].set_xlabel(r"$\sigma_{ppol/r²}$")
ax_dict["std_case"].set_ylabel("Range [m]")
ax_dict["std_cal"].scatter(
    df_std.sel(internal_temperature_bins=18).ppol_r,
    df_std.range,
    s=1,
)

filtered = noise_filter_std(df_std.sel(internal_temperature_bins=18).ppol_r)
filtered[:100] = df_std.sel(internal_temperature_bins=18).ppol_r[:100].values
ax_dict["std_cal"].scatter(
    filtered,
    df_std.range,
    s=1,
)
ax_dict["std_cal"].set_xlabel(r"$\sigma_{ppol/r²}$")
ax_dict["mean_case"].set_xlim(-1e-14, 3e-14)
ax_dict["mean_cal"].set_xlim(-1e-14, 3e-14)

ax_dict["std_case"].set_xlim(2e-14, 6e-14)
ax_dict["std_cal"].set_xlim(2e-14, 6e-14)

for n, ax_ in enumerate(ax_dict):
    ax_dict[ax_].text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_dict[ax_].transAxes,
        size=12,
    )
    if ax_ != "time":
        ax_dict[ax_].grid()

fig.savefig(
    "/media/viet/CL61/img/solar_example.png",
    dpi=600,
    bbox_inches="tight",
)

# %%
