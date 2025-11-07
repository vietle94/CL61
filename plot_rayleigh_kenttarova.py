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
    "2023-07-02": {
        "start_date": "2023-07-02",
        "end_date": "2023-07-03",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 5800,
        "z2": 6300,
        "zref": 6000,
    },
    "2023-11-10": {
        "start_date": "2023-11-10",
        "end_date": "2023-11-11",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 5300,
        "z2": 5800,
        "zref": 5600,
    },
    "2024-01-24": {
        "start_date": "2024-01-24",
        "end_date": "2024-01-25",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 5400,
        "z2": 5900,
        "zref": 5700,
    },
    "2024-03-25": {
        "start_date": "2024-03-25",
        "end_date": "2024-03-26",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 5100,
        "z2": 5600,
        "zref": 5300,
    },
    "2024-05-08": {
        "start_date": "2024-05-08",
        "end_date": "2024-05-09",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 4500,
        "z2": 5000,
        "zref": 4800,
    },
    "2024-07-13": {
        "start_date": "2024-07-13",
        "end_date": "2024-07-14",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 4500,
        "z2": 5000,
        "zref": 4800,
    },
    "2024-09-15": {
        "start_date": "2024-09-15",
        "end_date": "2024-09-16",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 4200,
        "z2": 4700,
        "zref": 4500,
    },
    "2024-11-27": {
        "start_date": "2024-11-27",
        "end_date": "2024-11-28",
        "start_time": "20:00",
        "end_time": "04:00",
        "z1": 4300,
        "z2": 4800,
        "zref": 4500,
    },
}
my_date = my_date_dict["2024-11-27"]

# %%
df = xr.open_mfdataset(
    glob.glob(
        f"/media/viet/CL61/studycase/kenttarova/{my_date['start_date'].replace('-', '')}/*.nc"
    )
    + glob.glob(
        f"/media/viet/CL61/studycase/kenttarova/{my_date['end_date'].replace('-', '')}/*.nc"
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
        f"/media/viet/CL61/studycase/kenttarova/{my_date['start_date'].replace('-', '')}/weather/*ecmwf.nc"
    )
    + glob.glob(
        f"/media/viet/CL61/studycase/kenttarova/{my_date['end_date'].replace('-', '')}/weather/*ecmwf.nc"
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
z1 = 4300
z2 = 4800
zref = 4500

pnorm = df_mean["beta_att_smooth"] * (
    (mol_full_atten_mol).sel(range=slice(z1, z2)).mean()
    / df_mean["beta_att_smooth"].sel(range=slice(z1, z2)).mean()
)

# %%
lr = 50
aerosol_backscatter = backward(
    pnorm,
    mol_full,
    lr,
    zref,
    df_mean.range,
)
# %%
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
    f"/media/viet/CL61/img/klett_kenttarova_{my_date['start_date'].replace('-', '')}.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
