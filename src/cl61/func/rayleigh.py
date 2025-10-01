import numpy as np
from scipy.integrate import cumulative_trapezoid

"""
Bucholtz, A. Rayleigh-scattering calculations for the terrestrial atmosphere.
Applied Optics, Vol. 34, No. 15, 2766-2773 (1995)

https://doi.org/10.1364/AO.34.002765

"""


def molecular_backscatter(angle, temperature, pressure):
    """
    Calculate molecular backscatter coefficient at 910.55nm.

    Parameters:
        temperature (np.ndarray): Temperature in Kelvin [K]
        pressure (np.ndarray): Pressure in hPa [hPa]

    Returns:
        molBsc (np.ndarray): Molecular backscatter coefficient [km^-1 sr^-1]
    """
    y = 1.384e-2
    P_ray = 3 / (4 * (1 + 2 * y)) * ((1 + 3 * y) + (1 - y) * np.cos(angle) ** 2)
    molBsc = (
        1.5e-3 * (pressure / 1013.25) * (288.15 / temperature) / (4 * np.pi) * P_ray
    )
    return molBsc


""" Calculate atmospheric molecular depolarization ratio
Tomasi, C., Vitale, V., Petkov, B., Lupi, A. & Cacciari, A. Improved
algorithm for calculations of Rayleigh-scattering optical depth in standard
atmospheres. Applied Optics 44, 3320 (2005).

https://doi.org/10.1364/AO.44.003320

"""


def f1(wavelength):
    return 1.034 + 3.17 * 1e-4 * wavelength ** (-2)


def f2(wavelength):
    return 1.096 + 1.385 * 1e-3 * wavelength ** (-2) + 1.448 * 1e-4 * wavelength ** (-4)


def f(wavelength, C, water_over_air):
    """
    Calculate King's factor.

    Parameters:
        wavelength : wavelength [um]
        C: CO2 concentration [ppmv]
        water_over_air: Water vapor over air pressure ratio

    Returns:
        king_factor : King's factor
    """
    numerator = (
        0.78084 * f1(wavelength)
        + 0.20946 * f2(wavelength)
        + 0.00934 * 1
        + 1e-6 * C * 1.15
        + water_over_air * 1.001
    )
    denominator = 0.999640 + 1e-6 * C + water_over_air
    return numerator / denominator


def depo(king_factor):
    return (6 * king_factor - 6) / (7 * king_factor + 3)


def humidity_conversion(specific_humidity):
    """
    Calculate water over air pressure ratio from specific humidity.
    """
    return 1 / (18 / 29 * (1 / specific_humidity - 1) + 1)


def rayleigh_fitting(beta_profile, beta_mol, zmin, zmax):
    """
    Fit the backscatter profile to the molecular profile in a given height range.

    Parameters:
    - beta_profile: array-like
        Measured backscatter profile [m^-1 sr^-1].
    - beta_mol: array-like
        Molecular backscatter profile [m^-1 sr^-1].
    - zmin: float
        Minimum height for fitting [m].
    - zmax: float
        Maximum height for fitting [m].

    Returns:
    - popt: tuple
        Optimal parameters for the fitting.
    - pcov: 2D array
        Covariance of the optimal parameters.
    """
    # Select the fitting range
    beta_profile = beta_profile.sel(range=slice(zmin, zmax))
    beta_mol = beta_mol.sel(range=slice(zmin, zmax))

    # attenuated mol beta
    # 

    # Calibration factor
    # c = att_beta_mol.sum() / beta_profile.sum()
    c = beta_mol.sum() / beta_profile.sum()

    return c.values


def backward(ppol, mol_ppol, Sa, zref, z):
    ppol = ppol.sel(range=slice(None, zref))
    mol_ppol = mol_ppol.sel(range=slice(None, zref))
    z = z.sel(range=slice(None, zref))
    ppol = ppol[::-1]
    mol_ppol = mol_ppol[::-1]
    z = z[::-1]
    Zb = ppol * np.exp(
        2 * cumulative_trapezoid((Sa - 8 / 3 * np.pi) * mol_ppol, z, initial=0)
    )
    Nb = (ppol[0] / mol_ppol[0]).values + 2 * cumulative_trapezoid(
        Sa * Zb, z, initial=0
    )

    beta_a = Zb / Nb - mol_ppol
    return beta_a[::-1]


def forward(ppol, mol_ppol, Sa, c, z):
    Zb = ppol * np.exp(
        -2 * cumulative_trapezoid((Sa - 8 / 3 * np.pi) * mol_ppol, z, initial=0)
    )
    Nb = c - 2 * cumulative_trapezoid(Sa * Zb, z, initial=0)

    beta_a = Zb / Nb - mol_ppol
    return beta_a
