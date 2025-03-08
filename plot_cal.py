import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib as mpl
import string
import numpy as np
import pywt


# %%
def noise_filter(profile, wavelet="bior1.5"):
    # wavelet transform
    level = 8
    n_pad = (len(profile) // 2**level + 4) * 2**level - len(profile)
    coeff = pywt.swt(
        np.pad(
            profile,
            (n_pad - n_pad // 2, n_pad // 2),
            "constant",
            constant_values=(0, 0),
        ),
        wavelet,
        trim_approx=True,
        # level=7,
        level=level,
    )

    uthresh = np.median(np.abs(coeff[1])) / 0.6745 * np.sqrt(2 * np.log(len(coeff[1])))
    # minimax_thresh = (
    #     np.median(np.abs(coeff[1]))
    #     / 0.6745
    #     * (0.3936 + 0.1829 * np.log2(len(coeff[1])))
    # )
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    filtered = pywt.iswt(coeff, wavelet)
    filtered = filtered[(n_pad - n_pad // 2) : len(profile) + (n_pad - n_pad // 2)]
    return filtered


# %%
for site, lim in zip(
    ["vehmasmaki", "hyytiala", "kenttarova"],
    [[-1e-14, 1e-14], [-1e-14, 1e-14], [-1e-14, 1e-14]],
):
    files = glob.glob(f"/media/viet/CL61/calibration/{site}/merged/*.nc")
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True, constrained_layout=True)
    for file in files:
        df = xr.open_dataset(file)
        ppol = df["p_pol"].mean(dim="time") / (df["range"]) ** 2
        xpol = df["x_pol"].mean(dim="time") / (df["range"]) ** 2
        ppol_d = noise_filter(ppol)
        xpol_d = noise_filter(xpol)
        ax[0].plot(
            ppol_d,
            ppol.range,
            ".",
            label=df.time.values[0].astype("datetime64[D]").astype(str),
        )
        ax[1].plot(
            xpol_d,
            xpol.range,
            ".",
            label=df.time.values[0].astype("datetime64[D]").astype(str),
        )
        ax[0].set_xlim(lim)
        ax[1].set_xlim([-5e-15, 5e-15])
        df.close()
    ax[0].set_xlabel("ppol/range^2")
    ax[1].set_xlabel("xpol/range^2")
    ax[0].set_ylabel("range")

    for ax_ in ax.flatten():
        ax_.grid()
        ax_.legend()
    fig.savefig(
        f"/media/viet/CL61/img/calibration_noise_{site}.png", dpi=600, bbox_inches="tight")

# %%
