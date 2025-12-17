import xarray as xr
import matplotlib.pyplot as plt
import glob
import string
import matplotlib.dates as mdates

myFmt = mdates.DateFormatter("%H:%M")

# %%
files_before = glob.glob("/media/viet/CL61/studycase/lindenberg/20240303/*.nc")
df_before_ = xr.open_mfdataset(files_before)

df_before = df_before_.sel(time=slice("2024-03-03 16:30:00", "2024-03-03 17:00:00"))

files_after = glob.glob("/media/viet/CL61/studycase/lindenberg/20241116/*.nc")
df_after_ = xr.open_mfdataset(files_after)
df_after = df_after_.sel(time=slice("2024-11-16 16:30:00", "2024-11-16 17:00:00"))

df_before = df_before.sel(range=slice(0, 4000))
df_after = df_after.sel(range=slice(0, 4000))

# %%
profile_before = (df_before["p_pol"]).mean(dim="time")
profile_after = (df_after["p_pol"]).mean(dim="time")

# %%
fig, ax = plt.subplots(
    1, 2, figsize=(7, 3), constrained_layout=True, sharey=True, sharex=True
)
ax[0].scatter(
    profile_before,
    profile_before.range,
    label="uncorrected",
    alpha=0.5,
    edgecolor="none",
    s=20,
)
ax[0].scatter(
    profile_before * 1.5,
    profile_before.range,
    label="corrected",
    alpha=0.5,
    edgecolor="none",
    s=20,
)
ax[0].set_title("2024-03-03", weight="bold")
ax[1].scatter(
    profile_after,
    profile_after.range,
    label="uncorrected",
    alpha=0.5,
    edgecolor="none",
    s=20,
)
ax[1].scatter(
    profile_after * 0.6,
    profile_after.range,
    label="corrected",
    alpha=0.5,
    edgecolor="none",
    s=20,
)
ax[1].set_title("2024-11-16", weight="bold")
ax[0].set_xlim([-5e-7, 2e-6])
for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.03,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
    ax_.legend()
    ax_.grid()
    ax_.set_xlabel("ppol [a.u.]")
fig.savefig(
    "/media/viet/CL61/img/studycase_lindenberg_corrected.png",
    dpi=600,
    bbox_inches="tight",
)
# %%
