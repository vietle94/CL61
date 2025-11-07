import xarray as xr
import matplotlib.pyplot as plt
import glob
from cl61.func import rayleigh
import numpy as np
from scipy.integrate import cumulative_trapezoid
import pandas as pd
from cl61.func.rayleigh import backward, forward
from scipy.linalg import lstsq
import string

# %%
df = xr.open_mfdataset(
    glob.glob("/media/viet/CL61/studycase/lindenberg/20230910/*.nc")
    + glob.glob("/media/viet/CL61/studycase/lindenberg/20230911/*.nc")
)
time_slice = slice("2023-09-10T20:00", "2023-09-11T04:00")
df = df.sel(time=time_slice)

df_mean = df.mean(dim="time", skipna=True)
df_mean = df_mean.sel(range=slice(100, None))
df_mean["beta_att_smooth"] = (
    ("range"),
    np.convolve(df_mean["beta_att"], np.repeat(1 / 30, 30), mode="same"),
)  # smoothing over 30 bins

# %%
aod = pd.read_csv(
    "/media/viet/CL61/studycase/lindenberg/20230101_20231231_MetObs_Lindenberg.lev20",
    skiprows=6,
)

aod["datetime"] = aod["Date(dd:mm:yyyy)"] + " " + aod["Time(hh:mm:ss)"]
aod["datetime"] = pd.to_datetime(aod["datetime"], format="%d:%m:%Y %H:%M:%S")
aod = aod.set_index("datetime")
###################################################################################
aod_ceilo = np.interp(
    910.55,
    np.array([870, 1020]),
    aod.loc["2023-09-11", ["AOD_870nm", "AOD_1020nm"]].iloc[0].values,
)
###################################################################################
# %%
model = xr.open_mfdataset(
    glob.glob("/media/viet/CL61/studycase/lindenberg/20230910/weather/*ecmwf.nc")
    + glob.glob("/media/viet/CL61/studycase/lindenberg/20230911/weather/*ecmwf.nc")
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
    (mol_full_atten_mol * (np.exp(-2 * aod_ceilo))).sel(range=slice(1500, None)),
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
        (mol_full_atten_mol * (np.exp(-2 * aod_ceilo))).sel(range=slice(z1, z2)).mean()
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
                    / (mol_full_atten_mol * (np.exp(-2 * aod_ceilo))).sel(
                        range=slice(z1, z2)
                    )
                ),
            )[0][0]
        )  # partical extinction coefficient, smaller, better (Baars, Pollynet 2016)
        b = lstsq(
            df_mean.range.sel(range=slice(z1, z2)).values.reshape(-1, 1),
            np.log(pnorm.sel(range=slice(z1, z2))),
        )[0][0]
        c = lstsq(
            df_mean.range.sel(range=slice(z1, z2)).values.reshape(-1, 1),
            np.log(mol_full_atten_mol * (np.exp(-2 * aod_ceilo))).sel(
                range=slice(z1, z2)
            ),
        )[0][0]
    except ValueError:
        continue
    print(z1, z2, a, np.abs(b - c))

# %%
z1 = 4000
z2 = 4500
zref = 4200
# %%
pnorm = df_mean["beta_att_smooth"] * (
    (mol_full_atten_mol * (np.exp(-2 * aod_ceilo))).sel(range=slice(z1, z2)).mean()
    / df_mean["beta_att_smooth"].sel(range=slice(z1, z2)).mean()
)

# %%
lr = 50
i = 0
while i < 10:
    aerosol_backscatter = backward(
        pnorm,
        mol_full,
        lr,
        zref,
        df_mean.range,
    )
    lr_new = aod_ceilo / (aerosol_backscatter.sum().values * 4.8)
    if np.abs(lr - lr_new) < 0.1:
        lr = lr_new
        break
    lr = lr_new
    print(lr)
    i += 1

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

# %%
fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
ax[0].plot(df_mean.beta_att, df_mean.range, label=r"$\beta'$")
ax[0].plot(
    (mol_full_atten_mol * (np.exp(-2 * aod_ceilo))).sel(range=slice(1500, None)),
    mol_full.sel(range=slice(1500, None)).height,
    label=r"$^\parallel\beta^'_{mol}$",
)
ax[1].plot(klett, klett.range)
ax[1].set_xlabel(r"$^\beta_{aerosol}$")
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
    "/media/viet/CL61/img/klett_Lindenberg_20230910.png", dpi=300, bbox_inches="tight"
)

# %%
