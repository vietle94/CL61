import pandas as pd
import matplotlib.pyplot as plt
import glob
import matplotlib.dates as mdates

# %%
fig, ax = plt.subplots(
    2, 2, figsize=(9, 4), sharex=True, sharey=True, constrained_layout=True
)
for site, ax_ in zip(
    ["vehmasmaki", "hyytiala", "kenttarova", "lindenberg"], ax.flatten()
):
    file_dir = f"/media/viet/CL61/{site}/Integration/"
    df = pd.concat(
        [pd.read_csv(f) for f in glob.glob(file_dir + "*.csv")], ignore_index=True
    )
    df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
    # ax_.plot(df["datetime"], df["integration"], ".", label="Integration_attrs")
    ax_.plot(df["datetime"], df["integration_t"], ".", label="Integration_cal")
    ax_.grid()
    ax_.legend()
    ax_.set_title(site.capitalize(), weight="bold")
    ax_.xaxis.set_major_locator(mdates.MonthLocator(6, 12))
    ax_.set_ylim([0, 15])
# fig.savefig("/media/viet/CL61/img/integration_attrs.png", dpi=600, bbox_inches="tight")
fig.savefig("/media/viet/CL61/img/integration_cal.png", dpi=600, bbox_inches="tight")
