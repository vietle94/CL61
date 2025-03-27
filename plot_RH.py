import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

file_dir = r"/media/viet/CL61/vehmasmaki/Diag/"


def process(f):
    df = pd.read_csv(f)
    if "internal_humidity" not in df.columns:
        return None
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["datetime"] > "2021-06-01"]
    rh = (
        df.set_index("datetime")
        .resample("1h")
        .mean(numeric_only=True)["internal_humidity"]
        .reset_index()
    )
    return rh


df = pd.concat([process(f) for f in glob.glob(file_dir + "*.csv")], ignore_index=True)

# %%
fig, ax = plt.subplots(figsize=(10, 3))
ax.scatter(df["datetime"], df["internal_humidity"], alpha=0.1, s=1)
ax.grid()
ax.set_ylabel("RH [%]")
fig.savefig("/media/viet/CL61/img/RH_vehmasmaki.png", dpi=600, bbox_inches="tight")

# %%
