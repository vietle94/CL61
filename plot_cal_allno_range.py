import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
from cl61.func.calibration_T import temperature_ref
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
    "vehmasmaki": [[-1.5e-7, 2.5e-7], [1e-20, 1e-9]],
    "hyytiala": [[-1.5e-7, 2.5e-7], [1e-20, 1e-9]],
    "kenttarova": [[-1.5e-7, 2.5e-7], [1e-20, 1e-9]],
}

plot_sub = {
    "vehmasmaki": ["a", "b", "c", "e"],
    "hyytiala": ["f", "g", "h", "i"],
    "kenttarova": ["j", "k", "l", "m"],
}
# %%
fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharey=True)
for t, color in zip(t_valid, my_c):
    ax[0, 0].plot(
        df_mean.sel(internal_temperature_bins=t).ppol_ref * df_mean.range**2,
        df_mean.range,
        color=color,
        label=t,
    )
    ax[0, 1].plot(
        df_std.sel(internal_temperature_bins=t).ppol_ref ** 2 * df_mean.range**4,
        df_mean.range,
        color=color,
        label=t,
    )

    ax[1, 0].plot(
        df_mean.sel(internal_temperature_bins=t).xpol_ref * df_mean.range**2,
        df_mean.range,
        color=color,
        label=t,
    )

    ax[1, 1].plot(
        df_std.sel(internal_temperature_bins=t).xpol_ref ** 2 * df_mean.range**4,
        df_mean.range,
        color=color,
        label=t,
    )

handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="lower center", ncol=6, title="Internal Temperature [°C]"
)
ax[0, 0].set_xlim(plot_lim[site][0])
ax[1, 0].set_xlim(plot_lim[site][0])
ax[0, 1].set_xlim(plot_lim[site][1])
ax[1, 1].set_xlim(plot_lim[site][1])
ax[0, 1].set_xscale("log")
ax[1, 1].set_xscale("log")
# ax[0, 0].set_xlim(-1.5e-7, 2.5e-7)
# ax[1, 0].set_xlim(-1.5e-7, 2.5e-7)

ax[0, 0].set_xlabel(r"$\mu_{ppol}$ [a.u.]")
ax[0, 1].set_xlabel(r"$\sigma²_{ppol}$ [a.u.]")
ax[1, 0].set_xlabel(r"$\mu_{xpol}$ [a.u.]")
ax[1, 1].set_xlabel(r"$\sigma²_{xpol}$ [a.u.]")

ax[0, 0].set_ylabel("Range [m]")
ax[1, 0].set_ylabel("Range [m]")
fig.subplots_adjust(bottom=0.3, hspace=0.5)

for ax_, lab in zip(ax.flatten(), plot_sub[site]):
    ax_.text(
        -0.0,
        1.03,
        "(" + lab + ")",
        transform=ax_.transAxes,
        size=12,
    )
    ax_.xaxis.get_offset_text().set_fontsize(8)

    ax_.grid()
fig.suptitle(site.capitalize(), y=0.98, weight="bold")
fig.savefig(
    f"/media/viet/CL61/img/calibration_all_no_range_{site}.png",
    dpi=600,
    bbox_inches=None,
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
    "vehmasmaki": np.array([[-0.1e-9, 0.5e-8], [1e-21, 1e-14]]),
    "kenttarova": np.array([[-0.1e-9, 0.5e-8], [1e-21, 1e-14]]),
    "hyytiala": np.array([[-0.1e-9, 0.5e-8], [1e-21, 1e-14]]),
}

# %%
fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharey=True)
for t, color in zip(t_valid, my_c):
    ax[0, 0].plot(
        df_mean.sel(internal_temperature_bins=t).ppol_ref * df_mean.range**2,
        df_mean.range,
        color=color,
        label=t,
    )
    ax[0, 1].plot(
        df_std.sel(internal_temperature_bins=t).ppol_ref ** 2 * df_mean.range**4,
        df_mean.range,
        color=color,
        label=t,
    )

    ax[1, 0].plot(
        df_mean.sel(internal_temperature_bins=t).xpol_ref * df_mean.range**2,
        df_mean.range,
        color=color,
        label=t,
    )

    ax[1, 1].plot(
        df_std.sel(internal_temperature_bins=t).xpol_ref ** 2 * df_mean.range**4,
        df_mean.range,
        color=color,
        label=t,
    )

handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="lower center", ncol=6, title="Internal Temperature [°C]"
)
ax[0, 0].set_xlim(plot_lim[site][0, :])
ax[1, 0].set_xlim(plot_lim[site][0, :])
# ax[0, 0].set_xlim(-0.1e-9, 0.5e-8)
# ax[1, 0].set_xlim(-0.1e-9, 0.5e-8)
ax[0, 1].set_xlim(plot_lim[site][1, :])
ax[1, 1].set_xlim(plot_lim[site][1, :])
ax[0, 1].set_xscale("log")
ax[1, 1].set_xscale("log")

ax[0, 0].set_xlabel(r"$\mu_{ppol}$ [a.u.]")
ax[0, 1].set_xlabel(r"$\sigma²_{ppol}$ [a.u.]")
ax[1, 0].set_xlabel(r"$\mu_{xpol}$ [a.u.]")
ax[1, 1].set_xlabel(r"$\sigma²_{xpol}$ [a.u.]")

ax[0, 0].set_ylim([50, 1000])
ax[0, 0].set_ylabel("Range [m]")
ax[1, 0].set_ylabel("Range [m]")
fig.subplots_adjust(bottom=0.3, hspace=0.5)

for ax_, lab in zip(ax.flatten(), plot_sub[site]):
    ax_.text(
        -0.0,
        1.03,
        "(" + lab + ")",
        transform=ax_.transAxes,
        size=12,
    )
    ax_.grid()
    ax_.xaxis.get_offset_text().set_fontsize(8)
    # ax_.set_xscale("log")

fig.suptitle(site.capitalize(), y=0.98, weight="bold")
fig.savefig(
    f"/media/viet/CL61/img/calibration_1k_no_range_{site}.png",
    dpi=600,
    bbox_inches=None,
)
# %%
