def calibration_etaS(profile):
    b_integrated = 5 * profile["p_pol"].sum(dim="range")
    etaS = 1 / (2 * b_integrated)
    return etaS
