import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from cl61.func.calibration_T import temperature_ref
import glob
import string
from cl61.func import rayleigh
import numpy as np
from matplotlib.ticker import FuncFormatter

# %%
site = "kenttarova"
files = glob.glob(f"/media/viet/CL61/calibration/{site}/merged/*.nc")
df = xr.open_dataset(files[2])
df_mean, df_std, _ = temperature_ref(df)

# %%
file_dir = "/media/viet/CL61/studycase/kenttarova/20240305/"
df_case_full = xr.open_mfdataset(file_dir + "*.nc")
df_case_full = df_case_full.sel(time=slice("2024-03-05T11:00", "2024-03-05T16:00"))
df_case = df_case_full.sel(time=slice("2024-03-05T11:00", "2024-03-05T13:00"))
df_case = df_case.isel(range=slice(1, None))
df_case["ppol_r"] = df_case["p_pol"] / (df_case["range"] ** 2)
df_case["xpol_r"] = df_case["x_pol"] / (df_case["range"] ** 2)
df_case_mean = df_case.mean(dim="time", skipna=True)
df_case_std = df_case.std(dim="time", skipna=True)
# profile = df_case.sel(time="2023-09-26T<10:00", method="nearest")

# %%
model = xr.open_dataset(
    "/media/viet/CL61/studycase/kenttarova/20240305/weather/20240305_kenttarova_ecmwf.nc"
)
model = model.sel(time=slice("2024-03-05T11:00", "2024-03-05T13:00")).mean(dim="time")
model = model.swap_dims({"level": "height"})
model = model[["temperature", "pressure", "q"]]
model = model.drop_vars("level")
model = model.sel(height=slice(None, 15000))
model = model.interp(height=df_mean.range)
mol_scatter = rayleigh.molecular_backscatter(
    2 * np.pi,
    model["temperature"],
    model["pressure"] / 100,  # Pa to hPa
)
mol_depo = rayleigh.depo(
    rayleigh.f(910.55, 380, rayleigh.humidity_conversion(model["q"]))
)
mol_ppol = mol_scatter / 1000 * (1 - mol_depo)
mol_xpol = mol_scatter / 1000 * mol_depo
# %%
fig = plt.figure(layout="constrained", figsize=(16, 9))
ax_dict = fig.subplot_mosaic(
    [
        ["time", "time", "time", "time"],
        ["mean_case", "mean_case_x", "mean_cal", "mean_cal_x"],
        ["std_case", "std_case_x", "std_cal", "std_cal_x"],
    ],
    sharey=True,
)
p = ax_dict["time"].pcolormesh(
    df_case_full["time"],
    df_case_full["range"],
    df_case_full["p_pol"].T,
    norm=LogNorm(vmin=1e-7, vmax=1e-4),
)
fig.colorbar(p, ax=ax_dict["time"], label=r"$ppol$")
ax_dict["time"].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax_dict["time"].xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax_dict["time"].set_ylabel("Range [km]")

ax_dict["mean_case"].scatter(
    df_case_mean.p_pol, df_case_mean.range, s=1, label=r"$\mu_{ppol}$"
)

ax_dict["mean_case"].scatter(
    mol_ppol,
    mol_ppol.height,
    s=1,
    label=r"$^\parallel\beta_{mol}$",
    c="C3",
)

ax_dict["mean_case"].set_xlabel(r"$\mu_{ppol}$")
ax_dict["mean_case"].set_ylabel("Range [km]")
ax_dict["mean_case"].legend(loc="upper right")
##################################################################

ax_dict["mean_case_x"].scatter(
    df_case_mean.x_pol, df_case_mean.range, s=1, label=r"$\mu_{xpol}$"
)
ax_dict["mean_case_x"].scatter(
    mol_xpol,
    mol_xpol.height,
    s=1,
    label=r"$^\perp\beta_{mol}/r²$",
    c="C3",
)
ax_dict["mean_case_x"].legend(loc="upper right")
ax_dict["mean_case_x"].set_xlabel(r"$\mu_{xpol}$")
#####################################################################

ax_dict["mean_cal"].scatter(
    df_mean.sel(internal_temperature_bins=18).p_pol,
    df_mean.range,
    s=1,
    c="C0",
)
ax_dict["mean_cal"].set_xlabel(r"$\mu_{ppol}$")
#####################################################################

ax_dict["mean_cal_x"].scatter(
    df_mean.sel(internal_temperature_bins=18).x_pol,
    df_mean.range,
    s=1,
    c="C0",
)
ax_dict["mean_cal_x"].set_xlabel(r"$\mu_{xpol}$")
#####################################################################

ax_dict["std_case"].scatter(df_case_std.ppol_r**2, df_case_std.range, s=1)
ax_dict["std_case"].set_xlabel(r"$\sigma²_{ppol}/r²$")
ax_dict["std_case"].set_ylabel("Range [km]")
#####################################################################

ax_dict["std_case_x"].scatter(df_case_std.xpol_r**2, df_case_std.range, s=1)
ax_dict["std_case_x"].set_xlabel(r"$\sigma²_{xpol}/r²$")
ax_dict["std_case_x"].set_ylabel("Range [km]")
#####################################################################

ax_dict["std_cal"].scatter(
    df_std.sel(internal_temperature_bins=18).ppol_r ** 2,
    df_std.range,
    s=1,
)

ax_dict["std_cal"].set_xlabel(r"$\sigma²_{ppol}/r²$")

#####################################################################
ax_dict["std_cal_x"].scatter(
    df_std.sel(internal_temperature_bins=18).xpol_r ** 2,
    df_std.range,
    s=1,
)
ax_dict["std_cal_x"].set_xlabel(r"$\sigma²_{xpol}/r²$")

#####################################################################

for x in ["mean_case", "mean_case_x", "mean_cal", "mean_cal_x"]:
    ax_dict[x].set_xlim(-5e-7, 5e-7)
# ax_dict["mean_case"].set_xlim(-5e-7, 5e-7)
# ax_dict["mean_case_x"].set_xlim(-5e-7, 5e-7)
# ax_dict["mean_cal"].set_xlim(-1e-14, 10e-14)
# ax_dict["mean_cal_x"].set_xlim(-1e-14, 10e-14)

for x in ["std_case", "std_case_x", "std_cal", "std_cal_x"]:
    ax_dict[x].set_xlim(4e-28, 36e-28)

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
    ax_dict[ax_].yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{x / 1000:.0f}")
    )
    ax_dict[ax_].set_ylim(0, 14000)

fig.savefig(
    "/media/viet/CL61/img/solar_example.png",
    dpi=600,
    bbox_inches="tight",
)

# %%
