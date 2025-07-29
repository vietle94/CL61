import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import string
from cl61.func.calibration_T import temperature_ref, noise_filter
import matplotlib as mpl

# %%
site = "kenttarova"
files = glob.glob(f"/media/viet/CL61/calibration/{site}/merged/*.nc")
df = xr.open_mfdataset(files)
df_mean, df_std, df_count = temperature_ref(df)

# %%
t_valid = df_count.internal_temperature_bins.where(
    (df_count.internal_temperature > 200).compute(), drop=True
).values

n1 = t_valid.size // 2
n2 = t_valid.size - n1
my_c = mpl.colormaps["RdBu_r"](
    np.append(np.linspace(0, 0.4, n1), np.linspace(0.6, 1, n2))
)
plot_lim = {
    "vehmasmaki": [5e-15, 1e-14],
    "hyytiala": [5e-15, 3e-14],
    "kenttarova": [2e-14, 8e-14],
}
# %%
fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharey=True)
for t, color in zip(t_valid, my_c):
    ax[0, 0].plot(
        df_mean.sel(internal_temperature_bins=t).ppol_ref,
        df_mean.range,
        color=color,
        label=t,
    )
    ax[0, 1].plot(
        df_std.sel(internal_temperature_bins=t).ppol_ref,
        df_mean.range,
        color=color,
        label=t,
    )

    ax[1, 0].plot(
        df_mean.sel(internal_temperature_bins=t).xpol_ref,
        df_mean.range,
        color=color,
        label=t,
    )

    ax[1, 1].plot(
        df_std.sel(internal_temperature_bins=t).xpol_ref,
        df_mean.range,
        color=color,
        label=t,
    )

handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5)
ax[0, 0].set_xlim([-5e-15, 1e-14])
ax[1, 0].set_xlim([-5e-15, 1e-14])
ax[0, 1].set_xlim(plot_lim[site])
ax[1, 1].set_xlim(plot_lim[site])

ax[0, 0].set_xlabel(r"$\mu_{ppol/r²}$")
ax[0, 1].set_xlabel(r"$\sigma_{ppol/r²}$")
ax[1, 0].set_xlabel(r"$\mu_{xpol/r²}$")
ax[1, 1].set_xlabel(r"$\sigma_{xpol/r²}$")

ax[0, 0].set_ylabel("Range [m]")
ax[1, 0].set_ylabel("Range [m]")
fig.subplots_adjust(bottom=0.3, hspace=0.5)

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
    f"/media/viet/CL61/img/calibration_all_{site}.png",
    dpi=600,
    bbox_inches="tight",
)
# %%
# 2000 m
###########################################
# site = "vehmasmaki"
# files = glob.glob(f"/media/viet/CL61/calibration/{site}/merged/*.nc")[:-1]
# df = xr.open_mfdataset(files)
# df_mean, df_std, df_count = temperature_ref(df)

# %%
t_valid = df_count.internal_temperature_bins.where(
    (df_count.internal_temperature > 200).compute(), drop=True
).values

n1 = t_valid.size // 2
n2 = t_valid.size - n1
my_c = mpl.colormaps["RdBu_r"](
    np.append(np.linspace(0, 0.4, n1), np.linspace(0.6, 1, n2))
)
plot_lim = {
    "vehmasmaki": np.array([[-5e-14, 1.5e-13], [0.5e-14, 2.5e-14]]),
    "kenttarova": np.array([[-2e-13, 1e-13], [2e-14, 8e-14]]),
    "hyytiala": np.array([[-7e-13, 1e-13], [5e-15, 3e-14]]),
}

# %%
fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharey=True)
for t, color in zip(t_valid, my_c):
    ax[0, 0].plot(
        df_mean.sel(internal_temperature_bins=t).ppol_ref,
        df_mean.range,
        color=color,
        label=t,
    )
    ax[0, 1].plot(
        df_std.sel(internal_temperature_bins=t).ppol_ref,
        df_mean.range,
        color=color,
        label=t,
    )

    ax[1, 0].plot(
        df_mean.sel(internal_temperature_bins=t).xpol_ref,
        df_mean.range,
        color=color,
        label=t,
    )

    ax[1, 1].plot(
        df_std.sel(internal_temperature_bins=t).xpol_ref,
        df_mean.range,
        color=color,
        label=t,
    )

handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5)
ax[0, 0].set_xlim(plot_lim[site][0, :])
ax[1, 0].set_xlim(plot_lim[site][0, :])
ax[0, 1].set_xlim(plot_lim[site][1, :])
ax[1, 1].set_xlim(plot_lim[site][1, :])

ax[0, 0].set_xlabel(r"$\mu_{ppol/r²}$")
ax[0, 1].set_xlabel(r"$\sigma_{ppol/r²}$")
ax[1, 0].set_xlabel(r"$\mu_{xpol/r²}$")
ax[1, 1].set_xlabel(r"$\sigma_{xpol/r²}$")

ax[0, 0].set_ylim([0, 1000])
ax[0, 0].set_ylabel("Range [m]")
ax[1, 0].set_ylabel("Range [m]")
fig.subplots_adjust(bottom=0.3, hspace=0.5)

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
    f"/media/viet/CL61/img/calibration_1k_{site}.png",
    dpi=600,
    bbox_inches="tight",
)
# %%
