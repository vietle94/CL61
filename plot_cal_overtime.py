import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import string
from cl61.func.calibration_T import temperature_ref, noise_filter

# %%
site = "vehmasmaki"
t_range = range(23, 25)
files = glob.glob(f"/media/viet/CL61/calibration/{site}/merged/*.nc")[:-1]
fig, ax = plt.subplots(2, 4, sharey="row", figsize=(9, 6), constrained_layout=True)
cmap = plt.get_cmap("tab10")
for ii, file in enumerate(files):
    df = xr.open_dataset(file)
    df_date = df.time.values[0].astype("datetime64[D]").astype(str)
    print(df_date)
    df["internal_temperature_floor"] = np.floor(df.internal_temperature)
    for i, t in enumerate(t_range):
        if t not in df["internal_temperature_floor"].values:
            continue
        df_ = df.where(df.internal_temperature_floor == t)
        df_mean, df_std, _ = temperature_ref(df_)
        df_plot = df_mean.sel(internal_temperature_bins=t)
        df_plot_std = df_std.sel(internal_temperature_bins=t)
        ax[0, i * 2].plot(df_plot.ppol_ref, df_plot.range, label=df_date, c=cmap(ii))
        ax[0, i * 2].set_xlim([-1e-14, 1e-14])
        ax[0, i * 2].set_title(f"T: {t}°C")

        ax[0, i * 2 + 1].plot(
            df_plot_std.ppol_ref, df_plot_std.range, label=df_date, c=cmap(ii)
        )
        ax[0, i * 2 + 1].set_xlim([0.5e-14, 3e-14])
        ax[0, i * 2 + 1].set_title(f"T: {t}°C")

        ax[1, i * 2].plot(df_plot.ppol_ref, df_plot.range, label=df_date, c=cmap(ii))
        ax[1, i * 2].set_xlim([-5e-14, 1.5e-13])
        ax[1, i * 2].set_ylim([0, 1000])
        ax[1, i * 2].set_title(f"T: {t}°C")

        ax[1, i * 2 + 1].plot(
            df_plot_std.ppol_ref, df_plot_std.range, label=df_date, c=cmap(ii)
        )
        ax[1, i * 2 + 1].set_xlim([0.5e-14, 3e-14])
        ax[1, i * 2 + 1].set_title(f"T: {t}°C")
        ax[1, i * 2 + 1].set_ylim([0, 1000])

    df.close()

for ax_ in ax.flatten():
    ax_.grid()
    ax_.legend(loc="upper right")

ax[0, 0].set_ylabel("Range [m]")
ax[1, 0].set_ylabel("Range [m]")

ax[1, 0].set_xlabel(r"$\mu_{ppol}/r^2$")
ax[1, 1].set_xlabel(r"$\sigma_{ppol}/r^2$")

ax[1, 2].set_xlabel(r"$\mu_{xpol}/r^2$")
ax[1, 3].set_xlabel(r"$\sigma_{xpol}/r^2$")

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )

fig.savefig(
    f"/media/viet/CL61/img/calibration_temperature_{site}_overtime.png",
    dpi=600,
    bbox_inches="tight",
)

# %%
