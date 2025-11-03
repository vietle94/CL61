import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.path as mpath


# %%
mat = np.array(
    [
        [0.5, 16.7674, 18.0927],
        [1.0, 15.5382, 17.4691],
        [1.5, 14.6756, 16.9479],
        [2.0, 14.0247, 16.5001],
        [2.5, 13.5116, 16.1084],
        [3.0, 13.095, 15.761],
        [3.5, 12.7492, 15.4498],
        [4.0, 12.4572, 15.1687],
        [4.5, 12.2074, 14.9129],
        [5.0, 11.9913, 14.6787],
        [5.5, 11.8025, 14.4634],
        [6.0, 11.6364, 14.2645],
        [6.5, 11.4892, 14.08],
        [7.0, 11.3578, 13.9083],
        [7.5, 11.2401, 13.7481],
        [8.0, 11.1341, 13.5981],
        [8.5, 11.0381, 13.4574],
        [9.0, 10.951, 13.3251],
        [9.5, 10.8716, 13.2004],
        [10.0, 10.799, 13.0827],
    ]
)
# %%
site = "vehmasmaki"
df = pd.concat(
    [pd.read_csv(x) for x in glob.glob(f"/media/viet/CL61/{site}/Cloud/*.csv")],
    ignore_index=True,
)
df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
df = df[df["datetime"] > pd.to_datetime("2000-01-01")]
df["range"] = (
    df["range"] + 76.8
)  # adjust for ref offset from the middle and flipped during convolution
df = df[df.range > 500]
df = df[df.range < 5000]
# df = df[df.cross_correlation > 3e-6]
df["month_year"] = df["datetime"].dt.to_period("M")
# march = df[df.datetime.dt.month == 5]

# %%
polygon_x = np.r_[mat[:, 1], mat[::-1, 2], mat[0, 1]]
polygon_y = np.r_[mat[:, 0] * 1000, mat[::-1, 0] * 1000, mat[0, 0] * 1000]
polygon = mpath.Path(np.c_[polygon_x, polygon_y])

# %%
filtered = df.groupby(df.month_year).filter(lambda x: len(x) > 500)
filtered_grouped = filtered.groupby(filtered.month_year)
n_group = len(filtered_grouped)
fig, axes = plt.subplots(
    (n_group + 6 - 1) // 6,
    6,
    figsize=(16, 9),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
calibration_factor = []
calibration_time = []
c_range = np.arange(0.5, 10 + 0.01, 0.01)
for (id, grp), ax in zip(filtered_grouped, axes.flatten()):
    if site == "vehamsmaki":
        if id in [
            pd.Period("2023-07"),
            pd.Period("2023-08"),
            pd.Period("2023-09"),
            pd.Period("2024-11"),
            pd.Period("2024-12"),
        ]:
            grp = grp[grp.cross_correlation > 3e-7]
        else:
            grp = grp[grp.cross_correlation > 7e-7]

    elif site == "hyytiala":
        if id in [
            pd.Period("2024-02"),
            pd.Period("2024-03"),
        ]:
            grp = grp.drop(
                grp[
                    (grp.datetime > pd.to_datetime("2024-02-15"))
                    & (grp.datetime < pd.to_datetime("2024-03-14"))
                ].index
            )
            grp = grp[grp.cross_correlation > 5e-7]
        elif id in [
            pd.Period("2024-11"),
            pd.Period("2024-12"),
        ]:
            grp = grp[grp.cross_correlation > 5e-7]
        else:
            grp = grp[grp.cross_correlation > 5e-7]
    elif site == "kenttarova":
        grp = grp[grp.cross_correlation > 5e-7]
    elif site == "lindenberg":
        grp = grp[grp.cross_correlation > 5e-7]
    temp = []
    for x in c_range:
        polygon_check = polygon.contains_points(np.c_[grp.etaS / x, grp.range])
        temp.append(np.sum(polygon_check) / len(polygon_check))
    # fig, ax = plt.subplots()
    # ax.plot(np.linspace(0.5, 2, 50), temp)
    # ax.grid()
    c = c_range[np.argmax(temp)]
    print(c)
    calibration_factor.append(c)
    calibration_time.append(id.to_timestamp())

    bin_range = np.linspace(500, 5000, 50)
    bin_etaS = np.linspace(0, 20, 50)
    p = ax.hist2d(grp.etaS / c, grp.range, bins=[bin_etaS, bin_range], cmin=1)
    fig.colorbar(p[3], ax=ax, label="Counts")
    ax.plot(mat[:, 1], mat[:, 0] * 1000, color="red")
    ax.plot(mat[:, 2], mat[:, 0] * 1000, color="red")
    ax.set_xlim(0, 20)
    ax.set_ylim(500, None)
    ax.grid()
    ax.set_title(id.strftime("%Y-%m"))
    ax.text(0.1, 0.8, f"c={c:.2f}", transform=ax.transAxes, color="red", weight="bold")

for ax_ in axes[-1, :]:
    ax_.set_xlabel(r"$\eta S$")
for ax_ in axes[:, 0]:
    ax_.set_ylabel("Cloud height (m)")

# %%
fig.savefig(
    f"/media/viet/CL61/img/cloud_calibration_{site}.png",
    dpi=600,
    bbox_inches="tight",
)
# %%
pd.DataFrame(
    {
        "datetime": calibration_time,
        "site": site,
        "calibration_factor": calibration_factor,
    }
).to_csv(
    f"/media/viet/CL61/cloud_calibration/cloud_calibration_{site}.csv", index=False
)

# %%
fig, ax = plt.subplots()
ax.plot(calibration_time, calibration_factor, marker="o")
ax.grid()
ax.set_ylabel("Calibration factor")
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m\n%Y"))
# %%
