import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
from cl61.func.calibration_cloud import calibration_etaS
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
import string
import pandas as pd
import h5py

myFmt = mdates.DateFormatter("%H:%M")
# %%
file_dir = "/media/viet/CL61/cloud_calibration/"
mat = h5py.File(glob.glob(file_dir + "Hyytiala.mat")[0])
df = pd.DataFrame(np.array(mat.get("Hyytiala")).T)
df.columns = [
    "datetime",
    "cloud_base",
    "cloud_top",
    "beta",
    "etaS",
]
# %%
pd.to_datetime(df["datetime"])
df["datetime"] = pd.to_datetime(df["datetime"] - 719529, unit="D")
