import numpy as np
import glob
import pandas as pd
import h5py


def calibration_etaS(profile):
    b_integrated = 5 * profile["p_pol"].sum(dim="range")
    etaS = 1 / (2 * b_integrated)
    return etaS


def cloud_calibration(site):
    site_name = {
        "vehmasmaki": "Vehmasmaki",
        "hyytiala": "Hyytiala",
        "kenttarova": "Kenttarova",
        "lindenberg": "Lindeberg",
    }
    mat = h5py.File(
        glob.glob(f"/media/viet/CL61/cloud_calibration/{site_name[site]}.mat")[0]
    )
    df = pd.DataFrame(np.array(mat.get(site_name[site])).T)
    df.columns = [
        "datetime",
        "cloud_base",
        "cloud_top",
        "beta",
        "etaS",
    ]
    df["datetime"] = pd.to_datetime(df["datetime"] - 719529, unit="D")
    df["c"] = (1 / (2 * df["etaS"])) / df["beta"]
    return df
