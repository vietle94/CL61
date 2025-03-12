import glob
import xarray as xr
import matplotlib.pyplot as plt
import string
from cl61.func.calibration_T import temperature_ref

files = glob.glob("/media/viet/CL61/calibration/kenttarova/merged/*.nc")
df = xr.open_mfdataset(files)
df["ppol_r"] = df.p_pol / (df.range**2)
df["xpol_r"] = df.x_pol / (df.range**2)
df_mean, df_std = temperature_ref(df)

# %%
fig, ax = plt.subplots(1, 3, figsize=(9, 4), constrained_layout=True, sharey=True)
for ax_, t in zip(ax.flatten(), [10, 20, 30]):
    ax_.plot(
        df_mean["ppol_r"].sel(internal_temperature_bins=t),
        df_mean["range"],
        label="original",
    )
    ax_.plot(
        df_mean["ppol_ref"].sel(internal_temperature_bins=t),
        df_mean["range"],
        label="filtered",
    )
    ax_.legend(loc="upper right")
    ax_.set_xlabel("ppol/r^2")
    ax_.set_xlim([-1e-14, 1e-14])
    ax_.grid()

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.03,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
fig.savefig("/media/viet/CL61/img/wavelet_example.png", dpi=600, bbox_inches="tight")
