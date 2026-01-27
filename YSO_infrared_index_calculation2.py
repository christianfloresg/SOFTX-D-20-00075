from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import astropy.units as u
from dust_extinction.grain_models import WD01

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
# Fit alpha between 2–24 um (generic columns)
# ----------------------------
def fit_alpha_between_2_and_24_microns(
    df: pd.DataFrame,
    flux_col: str = "lamFlam",
    err_col: str = "e_lamFlam",
) -> tuple[float, float, float, float, int]:
    """
    Weighted fit in log10-space:
      log10(flux) = alpha*log10(lam) + b

    Returns:
      alpha, alpha_unc, b, b_unc, n_points_used
    """
    lam = df["lam_m"].to_numpy(dtype=float)
    flux = df[flux_col].to_numpy(dtype=float)
    err = df[err_col].to_numpy(dtype=float)

    lo = 2e-6
    hi = 24e-6
    m = (
        np.isfinite(lam)
        & np.isfinite(flux)
        & np.isfinite(err)
        & (lam >= lo)
        & (lam <= hi)
        & (flux > 0.0)
        & (err > 0.0)
    )

    lam = lam[m]
    flux = flux[m]
    err = err[m]
    n = lam.size
    if n < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"), int(n))

    x = np.log10(lam)
    y = np.log10(flux)

    sigma_y = (1.0 / math.log(10.0)) * (err / flux)
    w = 1.0 / sigma_y

    (alpha, b), cov = np.polyfit(x, y, deg=1, w=w, cov=True)
    alpha_unc = float(np.sqrt(cov[0, 0]))
    b_unc = float(np.sqrt(cov[1, 1]))
    return (float(alpha), float(alpha_unc), float(b), float(b_unc), int(n))


# ----------------------------
# De-redden (2–24 um only) using WD01('MWRV55')
# ----------------------------
def deredden_df_2_24um(
    df: pd.DataFrame,
    Av_mag: float,
) -> pd.DataFrame:
    """
    Adds columns:
      lamFlam_dered, e_lamFlam_dered

    De-reddening:
      trans = ext.extinguish(lam, Ebv)
      lamFlam_dered = lamFlam / trans
      e_lamFlam_dered = e_lamFlam / trans

    Notes:
    - WD01('MWRV55') uses Ebv, with Rv=5.5 for this model name.
    """
    df2 = df.copy()

    lam_m = df2["lam_m"].to_numpy(float)
    lam_um = lam_m * 1e6
    lamFlam = df2["lamFlam"].to_numpy(float)
    e_lamFlam = df2["e_lamFlam"].to_numpy(float)


    # only valid points; keep NaN elsewhere
    # good = np.isfinite(lam_um) & np.isfinite(lamFlam) & np.isfinite(e_lamFlam) & (lamFlam > 0) & (e_lamFlam >= 0)
    good = (lam_um<1.e4) & np.isfinite(lamFlam) & np.isfinite(e_lamFlam) & (lamFlam > 0) & (e_lamFlam >= 0)

    lamFlam_dered = np.full_like(lamFlam, np.nan, dtype=float)
    e_lamFlam_dered = np.full_like(e_lamFlam, np.nan, dtype=float)

    ext = WD01("MWRV55")
    Rv = 5.5
    Ebv = Av_mag / Rv

    print(Ebv)

    trans = ext.extinguish(lam_um[good] * u.um, Ebv=Ebv)#.value  # F_obs/F_int
    lamFlam_dered[good] = lamFlam[good] / trans
    e_lamFlam_dered[good] = e_lamFlam[good] / trans

    print(trans)

    df2["lamFlam_dered"] = lamFlam_dered
    df2["e_lamFlam_dered"] = e_lamFlam_dered
    return df2


# ----------------------------
# Plot (observed + dereddened + both fits)
# ----------------------------
def plot_lamFlam_with_fit(
    df: pd.DataFrame,
    source: str,
    alpha: float,
    alpha_unc: float,
    b: float,
    outdir: str | Path = "plots",
    show: bool = False,
    flux_col: str = "lamFlam",
    err_col: str = "e_lamFlam",
    label: str = "data",
    linestyle: str = "-",
) -> Path:
    """
    Plots one dataset (flux_col vs wavelength) + its 2–24um fitted line.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lam = df["lam_m"].to_numpy(dtype=float)
    flux = df[flux_col].to_numpy(dtype=float)
    err = df[err_col].to_numpy(dtype=float)

    m_all = (
        np.isfinite(lam) & np.isfinite(flux) & np.isfinite(err)
        & (lam > 0) & (flux > 0) & (err >= 0)
    )
    lam_um = lam[m_all] * 1e6
    flux = flux[m_all]
    err = err[m_all]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.errorbar(
        lam_um,
        flux,
        yerr=err,
        fmt="o",
        color="k" if flux_col == "lamFlam" else "tab:blue",
        ms=4,
        elinewidth=1,
        capsize=2,
        label=label,
    )

    if np.isfinite(alpha) and np.isfinite(b):
        lam_fit_um = np.logspace(np.log10(2.0), np.log10(24.0), 200)
        lam_fit_m = lam_fit_um * 1e-6
        flux_fit = 10 ** (alpha * np.log10(lam_fit_m) + b)
        ax.plot(lam_fit_um, flux_fit, linestyle, label=f"fit 2–24 µm (α={alpha:.3f} ± {alpha_unc:.3f})")

    ax.set_xlim(0.4, 1e3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Wavelength $\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$\lambda F_{\lambda}$ [W m$^{-2}$]")
    ax.set_title(source)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    safe = re.sub(r"[^A-Za-z0-9_.\-\[\]]+", "_", source)
    suffix = "obs" if flux_col == "lamFlam" else "dered"
    outpath = outdir / f"{safe}_{suffix}_lamFlam_fit.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return outpath


def plot_obs_and_dered_same_axes(
    df: pd.DataFrame,
    source: str,
    fit_obs: tuple[float, float, float],
    fit_dered: tuple[float, float, float] | None,
    outdir: str | Path = "plots",
    show: bool = False,
) -> Path:
    """
    One figure: observed + dereddened points, plus both fit lines.
    fit_obs   = (alpha, alpha_unc, b)
    fit_dered = (alpha, alpha_unc, b) or None
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lam = df["lam_m"].to_numpy(float)
    lam_um = lam * 1e6

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # observed
    m_obs = np.isfinite(lam_um) & np.isfinite(df["lamFlam"]) & np.isfinite(df["e_lamFlam"]) & (df["lamFlam"] > 0)
    ax.errorbar(
        lam_um[m_obs],
        df["lamFlam"].to_numpy(float)[m_obs],
        yerr=df["e_lamFlam"].to_numpy(float)[m_obs],
        fmt="o",
        ms=4,
        capsize=2,
        elinewidth=1,
        label="observed",
    )

    # dereddened
    if fit_dered is not None and "lamFlam_dered" in df.columns:
        m_d = np.isfinite(lam_um) & np.isfinite(df["lamFlam_dered"]) & np.isfinite(df["e_lamFlam_dered"]) & (df["lamFlam_dered"] > 0)
        ax.errorbar(
            lam_um[m_d],
            df["lamFlam_dered"].to_numpy(float)[m_d],
            yerr=df["e_lamFlam_dered"].to_numpy(float)[m_d],
            fmt="s",
            ms=4,
            capsize=2,
            elinewidth=1,
            label="dereddened",
        )

    # fit lines
    lam_fit_um = np.logspace(np.log10(2.0), np.log10(24.0), 200)
    lam_fit_m = lam_fit_um * 1e-6

    a, aunc, b = fit_obs
    if np.isfinite(a) and np.isfinite(b):
        ax.plot(lam_fit_um, 10 ** (a * np.log10(lam_fit_m) + b), "-", label=f"fit obs (α={a:.3f}±{aunc:.3f})")

    if fit_dered is not None:
        ad, adunc, bd = fit_dered
        if np.isfinite(ad) and np.isfinite(bd):
            ax.plot(lam_fit_um, 10 ** (ad * np.log10(lam_fit_m) + bd), "--", label=f"fit dered (α={ad:.3f}±{adunc:.3f})")

    ax.set_xlim(0.4, 1e3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Wavelength $\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$\lambda F_{\lambda}$ [W m$^{-2}$]")
    ax.set_title(source)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    safe = re.sub(r"[^A-Za-z0-9_.\-\[\]]+", "_", source)
    outpath = outdir / f"{safe}_obs_plus_dered.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return outpath

C_LIGHT = 299792458.0  # m/s
L_SUN = 3.828e26       # W (IAU 2015 nominal; fine for reporting)


def compute_tbol_fbol_from_df(df, flux_col="lamFlam", err_col=None, distance_pc=None, n_mc=0):
    """
    Compute Tbol and Fbol from an SED table with columns:
      lam_m (meters), flux_col = lambda*F_lambda (W/m^2)

    Parameters
    ----------
    df : pandas.DataFrame
    flux_col : str
        Column containing lambda*F_lambda (W/m^2)
    err_col : str or None
        Column with uncertainty in lambda*F_lambda (W/m^2). If provided and n_mc>0,
        uses Monte Carlo to estimate uncertainties on Tbol and Fbol.
    distance_pc : float or None
        If given, also returns Lbol (W and Lsun).
    n_mc : int
        If >0 and err_col provided, do Monte Carlo resampling for uncertainties.

    Returns
    -------
    dict with keys:
      Tbol_K, Fbol_Wm2, (optional) Lbol_W, Lbol_Lsun
      and if MC: Tbol_unc, Fbol_unc, (optional) Lbol_unc_W, Lbol_unc_Lsun
    """
    lam = np.asarray(df["lam_m"], dtype=float)
    lamFlam = np.asarray(df[flux_col], dtype=float)

    m = np.isfinite(lam) & np.isfinite(lamFlam) & (lam > 0) & (lamFlam > 0)
    lam = lam[m]
    lamFlam = lamFlam[m]

    if lam.size < 2:
        return {"Tbol_K": np.nan, "Fbol_Wm2": np.nan}

    # Convert to frequency space
    nu = C_LIGHT / lam                      # Hz
    Fnu = (lamFlam * lam) / C_LIGHT         # W m^-2 Hz^-1

    # Sort by increasing nu for integration
    idx = np.argsort(nu)
    nu = nu[idx]
    Fnu = Fnu[idx]

    # Integrals
    denom = np.trapz(Fnu, nu)               # ∫ Fnu dnu  -> Fbol
    numer = np.trapz(nu * Fnu, nu)          # ∫ nu Fnu dnu
    if denom <= 0:
        return {"Tbol_K": np.nan, "Fbol_Wm2": np.nan}

    nu_mean = numer / denom
    Tbol = 1.25e-11 * nu_mean
    Fbol = denom

    out = {"Tbol_K": float(Tbol), "Fbol_Wm2": float(Fbol)}

    # Optional Lbol
    if distance_pc is not None and np.isfinite(distance_pc):
        d_m = float(distance_pc) * 3.085677581e16
        Lbol = 4.0 * np.pi * d_m**2 * Fbol
        out["Lbol_W"] = float(Lbol)
        out["Lbol_Lsun"] = float(Lbol / L_SUN)

    # Optional Monte Carlo uncertainties
    if (n_mc > 0) and (err_col is not None) and (err_col in df.columns):
        e_lamFlam = np.asarray(df[err_col], dtype=float)[m][idx]
        e_lamFlam = np.where(np.isfinite(e_lamFlam) & (e_lamFlam > 0), e_lamFlam, 0.0)

        # Draw lamFlam realizations; enforce positivity
        rng = np.random.default_rng(0)
        tbol_s = []
        fbol_s = []
        lbol_s = []

        lam_sorted = lam[idx]
        for _ in range(int(n_mc)):
            lamFlam_draw = rng.normal(lamFlam[idx], e_lamFlam)
            lamFlam_draw = np.where(lamFlam_draw > 0, lamFlam_draw, np.nan)

            Fnu_draw = (lamFlam_draw * lam_sorted) / C_LIGHT
            good = np.isfinite(Fnu_draw) & (Fnu_draw > 0)
            if good.sum() < 2:
                continue

            nu_g = nu[good]
            Fnu_g = Fnu_draw[good]
            denom_g = np.trapz(Fnu_g, nu_g)
            numer_g = np.trapz(nu_g * Fnu_g, nu_g)
            if denom_g <= 0:
                continue

            nu_mean_g = numer_g / denom_g
            tbol_s.append(1.25e-11 * nu_mean_g)
            fbol_s.append(denom_g)

            if distance_pc is not None and np.isfinite(distance_pc):
                lbol_s.append(4.0 * np.pi * d_m**2 * denom_g)

        if len(tbol_s) >= 5:
            out["Tbol_unc"] = float(np.nanstd(tbol_s, ddof=1))
            out["Fbol_unc"] = float(np.nanstd(fbol_s, ddof=1))
            if distance_pc is not None and np.isfinite(distance_pc) and len(lbol_s) >= 5:
                out["Lbol_unc_W"] = float(np.nanstd(lbol_s, ddof=1))
                out["Lbol_unc_Lsun"] = float(np.nanstd(np.asarray(lbol_s) / L_SUN, ddof=1))

    return out


# ----------------------------
# Batch run (now includes dereddened alpha)
# ----------------------------
def compute_alphas_in_directory(
    directory: str | Path,
    av_csv: str | Path | None = None,
    output_file: str | Path = "alpha_results.csv",
    make_plots: bool = True,
    plot_dir: str | Path = "plots",
    show_plots: bool = False,
) -> pd.DataFrame:
    """
    av_csv (optional): a file with columns like:
      source,Av
    where source matches infer_source_name() output.
    """
    directory = Path(directory)
    files = sorted(directory.glob("*.dat"))

    files=['Cleaned_sources_SEDS/[BHS98]MHO2_phot_cleaned_1.dat']
    # load Av table if provided
    av_map = {}
    if av_csv is not None:
        av_df = pd.read_csv(av_csv)

        # ensure numeric
        av_df["Av"] = pd.to_numeric(av_df["Av"], errors="coerce")
        av_df["Av_conn"] = pd.to_numeric(av_df["Av_conn"], errors="coerce")

        # choose Av if present, otherwise Av_conn
        av_df["Av_final"] = av_df["Av"].where(av_df["Av"].notna(), av_df["Av_conn"])

        # build lookup dictionary
        av_map = dict(
            zip(
                av_df["source"].astype(str),
                av_df["Av_final"].astype(float),
            )
        )


    rows = []
    for fp in files:
        source = infer_source_name(fp)
        try:
            df = read_yso_phot_dat(fp)

            # 1) observed alpha
            alpha, alpha_unc, b, b_unc, npts = fit_alpha_between_2_and_24_microns(df)


            # 2) dereddened alpha (only if Av exists)
            Av = av_map.get(source, np.nan)

            alpha_d = alpha_d_unc = b_d = b_d_unc = np.nan
            npts_d = 0

            df_for_plot = df


            if np.isfinite(Av):
                print('yes')
                df_for_plot = deredden_df_2_24um(df, Av_mag=float(Av))
                print(df_for_plot)
                alpha_d, alpha_d_unc, b_d, b_d_unc, npts_d = fit_alpha_between_2_and_24_microns(
                    df_for_plot, flux_col="lamFlam_dered", err_col="e_lamFlam_dered"
                )

                print(alpha_d)
            # 3) plotting
            plot_path = ""
            if make_plots:
                # one plot with both observed and dereddened if available
                fit_obs = (alpha, alpha_unc, b)
                fit_dered = (alpha_d, alpha_d_unc, b_d) if np.isfinite(Av) else None
                plot_path = str(
                    plot_obs_and_dered_same_axes(
                        df=df_for_plot,
                        source=source,
                        fit_obs=fit_obs,
                        fit_dered=fit_dered,
                        outdir=plot_dir,
                        show=show_plots,
                    )
                )

            # 4) compute Tbol and Fbol -> if distance is given.
            props_obs = compute_tbol_fbol_from_df(df, flux_col="lamFlam", err_col="e_lamFlam", n_mc=500)

            Tbol=props_obs['Tbol_K']
            Tbol_unc=props_obs['Tbol_unc']

            # 4) save row
            rows.append(
                {
                    "source": source,
                    "file": fp.name,
                    "Av_mag": Av,
                    "alpha_obs_2_24um": round(alpha,3),
                    "alpha_obs_unc": round(alpha_unc,3),
                    "alpha_dered_2_24um": round(alpha_d,3),
                    "alpha_dered_unc": round(alpha_d_unc,3),
                    "Tbol_K": round(Tbol,3),
                    "Tbol_unc":round(Tbol_unc,3),
                    "n_points_obs": npts,
                    "n_points_dered": npts_d,
                    "plot": plot_path,
                }
            )

        except Exception as e:
            rows.append(
                {
                    "source": source,
                    "file": fp.name,
                    "Av_mag": av_map.get(source, np.nan),
                    "alpha_obs_2_24um": float("nan"),
                    "alpha_obs_unc": float("nan"),
                    "alpha_dered_2_24um": float("nan"),
                    "alpha_dered_unc": float("nan"),
                    "Tbol_K": float("nan"),
                    "Tbol_unc": float("nan"),
                    "n_points_obs": 0,
                    "n_points_dered": 0,
                    "plot": "",
                    "error": str(e),
                }
            )

    out = pd.DataFrame(rows).sort_values(["source", "file"])
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    return out


if __name__ == "__main__":
    results = compute_alphas_in_directory(
        directory="Cleaned_sources_SEDS",
        av_csv="Av_table.csv",          # optional; comment out if you don't have it yet
        output_file="alpha_results_just_one.csv",
        make_plots=True,
        plot_dir="plots",
        show_plots=True,
    )

    print(results.to_string(index=False))
    # fp='Cleaned_sources_SEDS/[BHS98]MHO2_phot_cleaned_1.dat'
    # df = read_yso_phot_dat(fp)
    # df2 = deredden_df_2_24um(df,Av_mag=17.1)
    # print(df2)