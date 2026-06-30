from hapi import *
import numpy as np
import matplotlib.pyplot as plt

# %%
# --------------------------------------------------
# 1. Setup HITRAN
# --------------------------------------------------
db_begin("hitran")

lam0 = 910.55
sigma = 0.08  # 1-sigma Gaussian lidar spectrum

# --------------------------------------------------
# 2. Spectral range (just enough for your laser)
# --------------------------------------------------
lam_min = lam0 - 10
lam_max = lam0 + 10

nu_min = 1e7 / lam_max
nu_max = 1e7 / lam_min

fetch("H2O", 1, 1, nu_min, nu_max)

# %%
# --------------------------------------------------
# 3. Direct transmission spectrum (HAPI does physics)
# --------------------------------------------------
nu, coef = absorptionCoefficient_Voigt(SourceTables="H2O", HITRAN_units=False)
nu, trans = transmittanceSpectrum(nu, coef)

# convert to wavelength domain
lam = 1e7 / nu
idx = np.argsort(lam)

lam = lam[idx]
trans = trans[idx]

trans = trans**2  # 2-way transmission for lidar

# %%
fig, ax = plt.subplots()
ax.plot(lam, trans)

# %%
# --------------------------------------------------
# 4. Lidar spectral responses
# --------------------------------------------------
# lam0 = 910.55


def gaussian(sigma, lam0):
    g = np.exp(-0.5 * ((lam - lam0) / sigma) ** 2)
    return g / np.trapezoid(g, lam)


I_narrow = gaussian(0.08, 910.55)
I_broad = gaussian(1.44, 910)


# --------------------------------------------------
# 5. Instrument-averaged transmission
# --------------------------------------------------
def lidar_response(I):
    return np.trapezoid(trans * I, lam)


# --------------------------------------------------
# 6. Compare systems
# --------------------------------------------------
print("Narrow:", lidar_response(I_narrow))
print("Broad :", lidar_response(I_broad))
# %%
narrow = lidar_response(I_narrow)
broad = lidar_response(I_broad)

# %% Effect of water on narrow vs broad lidar
(1 - narrow) / (1 - broad)

# %%
