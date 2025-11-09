import xarray as xr
import matplotlib.pyplot as plt
import glob
from cl61.func import rayleigh
import numpy as np
from scipy.integrate import cumulative_trapezoid
from cl61.func.rayleigh import backward, forward
from scipy.linalg import lstsq
import string

# %%
my_date_dict = {
    "2022-06-24": {
        "start_date": "2022-06-24",
        "end_date": "2022-06-25",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 4200,
    },
    "2022-08-14": {
        "start_date": "2022-08-14",
        "end_date": "2022-08-15",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 5100,
    },
    "2023-02-10": {
        "start_date": "2023-02-10",
        "end_date": "2023-02-11",
        "start_time": "20:00",
        "end_time": "02:00",
        "z1": 5600,
    },
    "2023-04-15": {
        "start_date": "2023-04-15",
        "end_date": "2023-04-16",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 5100,
    },
    "2023-06-15": {
        "start_date": "2023-06-15",
        "end_date": "2023-06-16",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 4700,
    },
    "2023-08-07": {
        "start_date": "2023-08-07",
        "end_date": "2023-08-08",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 4100,
    },
    "2023-10-20": {
        "start_date": "2023-10-20",
        "end_date": "2023-10-21",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 4300,
    },
    "2024-03-18": {
        "start_date": "2024-03-18",
        "end_date": "2024-03-19",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 6300,
    },
    "2024-05-08": {
        "start_date": "2024-05-08",
        "end_date": "2024-05-09",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 6600,
    },
    "2024-06-22": {
        "start_date": "2024-06-22",
        "end_date": "2024-06-23",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 5400,
    },
    "2024-07-21": {
        "start_date": "2024-07-21",
        "end_date": "2024-07-22",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 5000,
    },
    "2024-09-04": {
        "start_date": "2024-09-04",
        "end_date": "2024-09-05",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 4500,
    },
}
my_date = my_date_dict["2024-11-09"]

# %%
df = xr.open_mfdataset(
    glob.glob(
        f"/media/viet/CL61/studycase/vehmasmaki/{my_date['start_date'].replace('-', '')}/*.nc"
    )
    + glob.glob(
        f"/media/viet/CL61/studycase/vehmasmaki/{my_date['end_date'].replace('-', '')}/*.nc"
    )
)
time_slice = slice(
    f"{my_date['start_date']}T{my_date['start_time']}",
    f"{my_date['end_date']}T{my_date['end_time']}",
)
df = df.sel(time=time_slice)

df_mean = df.mean(dim="time", skipna=True)
df_mean = df_mean.sel(range=slice(100, None))
df_mean["beta_att_smooth"] = (
    ("range"),
    np.convolve(df_mean["beta_att"], np.repeat(1 / 30, 30), mode="same"),
)  # smoothing over 30 bins

# %%
model = xr.open_mfdataset(
    glob.glob(
        f"/media/viet/CL61/studycase/vehmasmaki/{my_date['start_date'].replace('-', '')}/weather/*.nc"
    )
    + glob.glob(
        f"/media/viet/CL61/studycase/vehmasmaki/{my_date['end_date'].replace('-', '')}/weather/*.nc"
    )
)
model = model.sel(time=time_slice).mean(dim="time")

model = model.swap_dims({"level": "height"})
model = model[["temperature", "pressure", "q"]]
model = model.drop_vars("level")
model = model.interp(height=df_mean.range)
mol_scatter = rayleigh.molecular_backscatter(
    2 * np.pi,
    model["temperature"],
    model["pressure"] / 100,  # Pa to hPa
)

mol_full = mol_scatter / 1000
mol_full_atten_mol = mol_full * np.exp(
    -2 * cumulative_trapezoid(mol_full * 8 / 3 * np.pi, mol_full.height, initial=0)
)

# %%
# SNR check
noise_std = (df.beta_att / (df.range**2)).isel(range=slice(-300, None)).std()
snr = (df_mean["beta_att_smooth"] / (df.range**2)) / noise_std
h = (snr > 0.05).idxmin().values

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
ax[0].plot(df_mean.beta_att, df_mean.range, label="beta_att")
ax[0].plot(
    (mol_full_atten_mol).sel(range=slice(1500, None)),
    mol_full_atten_mol.sel(range=slice(1500, None)).height,
    label=r"$^\parallel\beta^'_{mol}$",
)
ax[0].set_xlabel(r"$\beta$")
ax[1].plot(snr, df_mean.range)
ax[1].axhline(h, color="red", linestyle="--", label=f"SNR=0.05\n at {h} m")
ax[1].set_xlim(-0.01, 1)
ax[0].set_xlim(-3e-7, 3e-7)
ax[1].set_xlabel("SNR")
ax[0].set_ylabel("Height [m]")
ax[0].set_ylim(None, 14000)
for ax_ in ax:
    ax_.grid()
    ax_.legend()
# fig.savefig(
#     "/media/viet/CL61/img/snr_Lindenberg_20240316.png", dpi=300, bbox_inches="tight")

# %%
for z1 in np.arange(4000, h + 500, 100):
    z2 = z1 + 500
    pnorm = df_mean.beta_att_smooth * (
        (mol_full_atten_mol).sel(range=slice(z1, z2)).mean()
        / df_mean.beta_att_smooth.sel(range=slice(z1, z2)).mean()
    )
    try:
        a = (
            -1
            / 2
            * lstsq(
                df_mean.range.sel(range=slice(z1, z2)).values.reshape(-1, 1),
                np.log(
                    pnorm.sel(range=slice(z1, z2))
                    / (mol_full_atten_mol).sel(range=slice(z1, z2))
                ),
            )[0][0]
        )  # partical extinction coefficient, smaller, better (Baars, Pollynet 2016)
        b = lstsq(
            df_mean.range.sel(range=slice(z1, z2)).values.reshape(-1, 1),
            np.log(pnorm.sel(range=slice(z1, z2))),
        )[0][0]
        c = lstsq(
            df_mean.range.sel(range=slice(z1, z2)).values.reshape(-1, 1),
            np.log(mol_full_atten_mol).sel(range=slice(z1, z2)),
        )[0][0]
    except ValueError:
        continue
    print(z1, z2, a, np.abs(b - c))

# %%
z1 = 4500
z2 = z1 + 500
zref = (z1 + z2) / 2

pnorm = df_mean["beta_att_smooth"] * (
    (mol_full_atten_mol).sel(range=slice(z1, z2)).mean()
    / df_mean["beta_att_smooth"].sel(range=slice(z1, z2)).mean()
)


lr = 50
aerosol_backscatter = backward(
    pnorm,
    mol_full,
    lr,
    zref,
    df_mean.range,
)

c = (
    1
    / lstsq(
        (
            (
                mol_full.sel(range=slice(100, zref))
                + aerosol_backscatter.sel(range=slice(100, zref))
            )
            * np.exp(
                -2
                * cumulative_trapezoid(
                    lr * aerosol_backscatter.sel(range=slice(100, zref))
                    + (mol_full.sel(range=slice(100, zref)) * 8 / 3 * np.pi),
                    df_mean.range.sel(range=slice(100, zref)),
                    initial=0,
                )
            )
        ).values.reshape(-1, 1),
        df_mean["beta_att_smooth"].sel(range=slice(100, zref)).values.reshape(-1, 1),
    )[0][0]
)
print(c)
# %%
klett = forward(df_mean.beta_att, mol_full, lr, 1 / c, df_mean.range)

fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
ax[0].plot(df_mean.beta_att, df_mean.range, label=r"$\beta'$")
ax[0].plot(
    (mol_full_atten_mol).sel(range=slice(1500, None)),
    mol_full.sel(range=slice(1500, None)).height,
    label=r"$^\parallel\beta^'_{mol}$",
)
ax[1].plot(klett, klett.range)
ax[1].set_xlabel(r"$\beta_{aerosol}$")
ax[0].set_ylabel("Height [m]")
ax[0].set_ylim(None, 14000)
ax[0].set_xlabel(r"$\beta^'$")
ax[0].legend()

for n, ax_ in enumerate(ax.flatten()):
    ax_.text(
        -0.0,
        1.05,
        "(" + string.ascii_lowercase[n] + ")",
        transform=ax_.transAxes,
        size=12,
    )
    ax_.grid()

fig.suptitle(f"c = {c[0]:.2f}")
fig.savefig(
    f"/media/viet/CL61/img/klett_vehmasmaki_{my_date['start_date'].replace('-', '')}.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
