import numpy as np
import pandas as pd
import glob
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import string
from cl61.func.study_case import process_raw, background_noise
from matplotlib.colors import LogNorm
import matplotlib.ticker as mtick
from cl61.func import rayleigh

myFmt = mdates.DateFormatter("%Y\n%m-%d\n%H:%M")
def slice_to_title(s: slice) -> str:
    start = s.start.replace("T", " ")
    stop = s.stop.replace("T", " ")
    return f"{start} → {stop}"

# %%
date = "20240326"
time_slice = slice("2024-03-26T08:00", "2024-03-26T12:00")
model = xr.open_dataset(
    glob.glob(f"/media/viet/CL61/studycase/kenttarova/{date}/weather/*ecmwf.nc")[0]
)

model = model.sel(time=time_slice).mean(dim="time")

model = model.swap_dims({"level": "height"})
model = model[["temperature", "pressure", "q"]]
model = model.drop_vars("level")

depo_mol = rayleigh.depo(
    rayleigh.f(0.91055, 425, rayleigh.humidity_conversion(model["q"]))
)

# %%
fig, ax = plt.subplots()
ax.plot(depo_mol, depo_mol.height)
ax.set_ylim(0, 15000)
ax.set_title(slice_to_title(time_slice))
fig.savefig(f"{date}.png", dpi=600, bbox_inches="tight")

# %%
