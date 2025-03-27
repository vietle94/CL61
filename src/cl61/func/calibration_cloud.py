def calibration_factor(profile):
    b_integrated = profile["p_pol"].sum(dim="range")
    c = 1 / (2 * b_integrated * 0.8 * 18.8)
    return c
