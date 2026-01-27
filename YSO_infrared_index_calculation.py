from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# I/O
# ----------------------------
def read_yso_phot_dat(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)

    header_idx = None
    with filepath.open("r") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.split()[0] == "lam":
                header_idx = i
                break
    if header_idx is None:
        raise ValueError(f"Could not find 'lam ...' header line in {filepath}")

    data_start = header_idx + 2

    colnames = [
        "lam_m",
        "band",
        "lamFlam",
        "e_lamFlam",
        "f_lamFlam",
        "u_lamFlam",
        "beam",
        "obsDate",
        "ref",
    ]

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        comment="#",
        header=None,
        names=colnames,
        skiprows=data_start,
        engine="python",
    )

    for c in ["lam_m", "lamFlam", "e_lamFlam"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def infer_source_name(filepath: str | Path) -> str:
    stem = Path(filepath).stem
    m = re.match(r"(.+?)_phot", stem)
    return m.group(1) if m else stem


# ----------------------------
# Fit alpha between 2–24 um
# ----------------------------
def fit_alpha_between_2_and_24_microns(
    df: pd.DataFrame,
) -> tuple[float, float, float, float, int]:
    """
    Returns:
      alpha, alpha_unc, intercept_b, intercept_unc, n_points_used

    Model in log10-space:
      y = log10(lamFlam) = alpha * x + b,  where x = log10(lam)
    """
    lam = df["lam_m"].to_numpy(dtype=float)
    lamFlam = df["lamFlam"].to_numpy(dtype=float)
    e_lamFlam = df["e_lamFlam"].to_numpy(dtype=float)

    lo = 2e-6
    hi = 24e-6
    m = (
        np.isfinite(lam)
        & np.isfinite(lamFlam)
        & np.isfinite(e_lamFlam)
        & (lam >= lo)
        & (lam <= hi)
        & (lamFlam > 0.0)
        & (e_lamFlam > 0.0)
    )

    lam = lam[m]
    lamFlam = lamFlam[m]
    e_lamFlam = e_lamFlam[m]
    n = lam.size
    if n < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"), int(n))

    x = np.log10(lam)
    y = np.log10(lamFlam)

    sigma_y = (1.0 / math.log(10.0)) * (e_lamFlam / lamFlam)
    w = 1.0 / sigma_y  # polyfit uses weights ~ 1/sigma

    (alpha, b), cov = np.polyfit(x, y, deg=1, w=w, cov=True)
    alpha_unc = float(np.sqrt(cov[0, 0]))
    b_unc = float(np.sqrt(cov[1, 1]))

    return (float(alpha), float(alpha_unc), float(b), float(b_unc), int(n))


# ----------------------------
# Plot
# ----------------------------
def plot_lamFlam_with_fit(
    df: pd.DataFrame,
    source: str,
    alpha: float,
    alpha_unc: float,
    b: float,
    outdir: str | Path = "plots",
    show: bool = False,
) -> Path:
    """
    Plots lambda*F_lambda vs lambda (both log scale), with errors, and overplots
    the best-fit single power-law between 2 and 24 microns.

    Saves PNG to outdir/<source>_lamFlam_fit.png (sanitized).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lam = df["lam_m"].to_numpy(dtype=float)
    lamFlam = df["lamFlam"].to_numpy(dtype=float)
    e_lamFlam = df["e_lamFlam"].to_numpy(dtype=float)

    # Keep finite, positive for log plot
    m_all = np.isfinite(lam) & np.isfinite(lamFlam) & np.isfinite(e_lamFlam) & (lam > 0) & (lamFlam > 0) & (e_lamFlam >= 0)
    lam = lam[m_all]
    lamFlam = lamFlam[m_all]
    e_lamFlam = e_lamFlam[m_all]

    lam_um = lam * 1e6  # x-axis in microns

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.errorbar(
        lam_um,
        lamFlam,
        yerr=e_lamFlam,
        fmt="o",
        color='k',
        ms=4,
        elinewidth=1,
        capsize=2,
        label="data",
    )

    # Overplot the 2–24 um fit if finite
    if np.isfinite(alpha) and np.isfinite(b):
        lam_fit_um = np.logspace(np.log10(2.0), np.log10(24.0), 200)
        lam_fit_m = lam_fit_um * 1e-6
        # y = log10(lamFlam) = alpha*log10(lam) + b
        lamFlam_fit = 10 ** (alpha * np.log10(lam_fit_m) + b)

        ax.plot(lam_fit_um, lamFlam_fit, "-", label=f"fit 2–24 µm (α={alpha:.3f} ± {alpha_unc:.3f})")

    ax.set_xlim(0.4,1e3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Wavelength $\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$\lambda F_{\lambda}$ [W m$^{-2}$]")
    ax.set_title(source)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    safe = re.sub(r"[^A-Za-z0-9_.\-\[\]]+", "_", source)
    outpath = outdir / f"{safe}_lamFlam_fit.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return outpath


# ----------------------------
# Batch run
# ----------------------------
def compute_alphas_in_directory(
    directory: str | Path,
    output_file: str | Path = "alpha_results.csv",
    make_plots: bool = True,
    plot_dir: str | Path = "plots",
    show_plots: bool = False,
) -> pd.DataFrame:
    directory = Path(directory)
    files = sorted(directory.glob("*.dat"))

    rows = []
    for fp in files:
        source = infer_source_name(fp)
        try:
            df = read_yso_phot_dat(fp)
            alpha, alpha_unc, b, b_unc, npts = fit_alpha_between_2_and_24_microns(df)

            plot_path = ""
            if make_plots:
                plot_path = str(
                    plot_lamFlam_with_fit(
                        df=df,
                        source=source,
                        alpha=alpha,
                        alpha_unc=alpha_unc,
                        b=b,
                        outdir=plot_dir,
                        show=show_plots,
                    )
                )

            rows.append(
                {
                    "source": source,
                    "file": fp.name,
                    "alpha_2_24um": alpha,
                    "alpha_unc": alpha_unc,
                    "intercept_b": b,
                    "intercept_unc": b_unc,
                    "n_points": npts,
                    "plot": plot_path,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "source": source,
                    "file": fp.name,
                    "alpha_2_24um": float("nan"),
                    "alpha_unc": float("nan"),
                    "intercept_b": float("nan"),
                    "intercept_unc": float("nan"),
                    "n_points": 0,
                    "plot": "",
                    "error": str(e),
                }
            )

    out = pd.DataFrame(rows).sort_values(["source", "file"])
    output_file = Path(output_file)
    out.to_csv(output_file, index=False)
    return out


if __name__ == "__main__":
    # Run in the folder containing your 33 .dat files
    results = compute_alphas_in_directory(
        directory="Cleaned_sources_SEDS",
        output_file="alpha_results.csv",
        make_plots=True,
        plot_dir="plots",
        show_plots=False,  # True if you want interactive pop-ups
    )
    print(results.to_string(index=False))
