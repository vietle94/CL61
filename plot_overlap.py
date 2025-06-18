import xarray as xr
import matplotlib.pyplot as plt
import glob

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)

files = glob.glob("/media/viet/CL61/studycase/vehmasmaki/20241209/*.nc")
df = xr.open_dataset(files[0])
ax[0].plot(df.overlap_function, df.range, label="vehmasmaki_2024")
ax[1].plot(df.overlap_function, df.range, label="vehmasmaki_2024")

files = glob.glob("/media/viet/CL61/studycase/vehmasmaki/20231201/*.nc")
df = xr.open_dataset(files[0])
ax[0].plot(df.overlap_function, df.range, label="vehmasmaki_2023")
ax[1].plot(df.overlap_function, df.range, label="vehmasmaki_2023")

files = glob.glob("/media/viet/CL61/studycase/hyytiala/20240801/*.nc")
df = xr.open_dataset(files[0])
ax[0].plot(df.overlap_function, df.range, label="hyytiala_2024")
ax[1].plot(df.overlap_function, df.range, label="hyytiala_2024")

files = glob.glob("/media/viet/CL61/studycase/kenttarova/20231002/*.nc")
df = xr.open_dataset(files[0])
ax[0].plot(df.overlap_function, df.range, label="kenttarova_2023")
ax[1].plot(df.overlap_function, df.range, label="kenttarova_2023")

ax[1].set_ylim(0, 600)
for ax_ in ax:
    ax_.legend()
    ax_.grid()
    ax_.set_xlabel("Overlap function")
    ax_.set_ylabel("Range [m]")
fig.savefig(
    "/media/viet/CL61/img/overlap_function.png",
    dpi=600,
    bbox_inches="tight",
)
