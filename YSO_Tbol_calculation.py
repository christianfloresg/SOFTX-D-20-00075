from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import astropy.units as u
from dust_extinction.grain_models import WD01

from YSO_infrared_index_calculation2 import read_yso_phot_dat

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

if __name__ == "__main__":

    fp='Cleaned_sources_SEDS/FPTau_phot_cleaned_1.dat'
    df = read_yso_phot_dat(fp)
    
    # props_obs = compute_tbol_fbol_from_df(df, flux_col="lamFlam", err_col="e_lamFlam", n_mc=500)
    props = compute_tbol_fbol_from_df(df, flux_col="lamFlam", err_col="e_lamFlam",
                                 distance_pc=140.0, n_mc=500)

    print(props['Tbol_K'])