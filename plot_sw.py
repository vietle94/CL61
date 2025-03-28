import pandas as pd
import matplotlib.pyplot as plt
import glob
import matplotlib.dates as mdates
import string

# %%
fig, ax = plt.subplots(
    2, 2, figsize=(9, 6), sharex=True, sharey=True, constrained_layout=True
)
for site, ax_ in zip(
    ["hyytiala", "vehmasmaki", "kenttarova", "lindenberg"], ax.flatten()
):
    file_dir = f"/media/viet/CL61/{site}/sw/"
    df = pd.concat(
        [pd.read_csv(f) for f in glob.glob(file_dir + "*.csv")], ignore_index=True
    )
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
    ax_.plot(df["datetime"], df["sw_version"], ".")
    ax_.xaxis.set_major_locator(mdates.MonthLocator(6, 12))
    ax_.grid()
    ax_.set_title(site.capitalize(), weight="bold")

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.03,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
ax[0, 0].set_ylabel("SW Version")
ax[1, 0].set_ylabel("SW Version")
fig.savefig("/media/viet/CL61/img/sw_version.png", dpi=600, bbox_inches="tight")

# %%
