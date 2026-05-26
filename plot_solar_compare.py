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
from scipy.integrate import cumulative_trapezoid

# %%
site = "kenttarova"
overlap = xr.open_dataset(
    glob.glob("/media/viet/CL61/studycase/kenttarova/20240915/*.nc")[0]
)["overlap_function"]

# %%
file_dir = "/media/viet/CL61/studycase/kenttarova/20240305/"
df_case_full = xr.open_mfdataset(file_dir + "*.nc")
# df_case_full = df_case_full.sel(time=slice("2024-03-05T00:00", "2024-03-05T02:00"))
df_case = df_case_full.sel(time=slice("2024-03-05T00:00", "2024-03-05T02:00"))
df_case = df_case.isel(range=slice(1, None))
df_case["ppol_r"] = df_case["p_pol"] / (df_case["range"] ** 2)
df_case_std = df_case.std(dim="time", skipna=True)

fig, ax = plt.subplots(1, 3, sharey=True, figsize=(9, 4))
ax[0].plot(df_case_std.ppol_r**2, df_case_std.range, ".")
ax[0].set_xlim(4e-28, 1e-26)
ax[0].set_ylabel("Range [km]")

df_case = df_case_full.sel(time=slice("2024-03-05T14:00", "2024-03-05T15:00"))
df_case = df_case.isel(range=slice(1, None))
df_case["ppol_r"] = df_case["p_pol"] / (df_case["range"] ** 2)
df_case_std = df_case.std(dim="time", skipna=True)

ax[1].plot(df_case_std.ppol_r**2, df_case_std.range, ".")
ax[1].set_xlim(4e-28, 1e-26)

file_dir = "/media/viet/CL61/studycase/kenttarova/20240306/"
df_case_full = xr.open_mfdataset(file_dir + "*.nc")
df_case = df_case_full.sel(time=slice("2024-03-06T20:00", "2024-03-06T22:00"))
df_case = df_case.isel(range=slice(1, None))
df_case["ppol_r"] = df_case["p_pol"] / (df_case["range"] ** 2)
df_case_std = df_case.std(dim="time", skipna=True)

ax[2].plot(df_case_std.ppol_r**2, df_case_std.range, ".")
ax[2].set_xlim(4e-28, 1e-26)

for n, ax_ in enumerate(ax):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
    ax_.set_ylim(0, 14000)
    ax_.grid()
    ax_.set_xlabel(r"$\sigma²_{ppol/r²}$ [a.u.]")
fig.savefig("/media/viet/CL61/img/solar_compare.png", dpi=600, bbox_inches="tight")

# %%
