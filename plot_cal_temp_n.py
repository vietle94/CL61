import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import string

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True, sharex=True)
xlim = [0, 0]
for ax, site in zip(axes.flatten(), ["vehmasmaki", "hyytiala", "kenttarova"]):
    files = glob.glob(f"/media/viet/CL61/calibration/{site}/merged/*.nc")
    temp_range = np.arange(-10, 40)
    bottom = np.zeros(temp_range[:-1].size)
    for i, file in enumerate(files):
        df = xr.open_dataset(file)
        plot_time = df.time.values[0].astype("datetime64[D]").astype(str)
        df_gr = df[["internal_temperature"]].groupby_bins(
            "internal_temperature", temp_range, labels=temp_range[:-1]
        )
        df_count = df_gr.count(dim="time")
        internal_temperature = np.nan_to_num(df_count.internal_temperature.values)
        p = ax.bar(
            df_count.internal_temperature_bins.values,
            internal_temperature,
            bottom=+bottom,
            label=plot_time,
        )
        bottom += np.nan_to_num(internal_temperature)
        df.close()
    ax.set_ylabel("Count")
    ax.set_xlabel("Internal temperature (°C)")
    ax.legend(
        bbox_to_anchor=(0.0, -0.3, 1.0, 0.102),
        loc="upper left",
        ncols=2,
        mode="expand",
        borderaxespad=0.0,
    )
    ax.grid()
    lim_index = (np.nonzero(bottom)[0][0] - 1, np.nonzero(bottom)[0][-1] + 1)
    new_xlim = [temp_range[x] for x in lim_index]
    xlim[0] = min(xlim[0], new_xlim[0])
    xlim[1] = max(xlim[1], new_xlim[1])
    ax.tick_params(axis="x", which="minor")
ax.set_xlim(xlim)
for n, ax_ in enumerate(axes.flatten()):
    ax_.text(
        -0.0,
        1.03,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
fig.savefig(
    "/media/viet/CL61/img/calibration_temperature.png", dpi=300, bbox_inches="tight"
)
