import numpy as np
import pandas as pd
from scipy import stats as spstats

def log_fit(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    t = 1.0 / np.log(x)
    slope, intercept, r_value, p_value, std_err = spstats.linregress(t, y)
    R2 = r_value**2
    residuals = y - (intercept + slope * t)
    return intercept, slope, R2, p_value, np.asarray(residuals)

def zero_term(L, zeros):
    total = 0.0
    for gamma in zeros:
        rho = 0.5 + 1j * gamma
        term = L**(rho + 1) / (rho * (rho + 1))
        total += term
    Z = (2 * np.log(L)**2 / L**2) * total.real
    return Z

def fit_with_zeros(df_sub, zeros):
    Lk = df_sub['interval_right'].values
    Rk = df_sub['Rk'].values
    t = 1.0 / np.log(Lk)
    Z = np.array([zero_term(L, zeros) for L in Lk])

    X = np.column_stack([np.ones_like(t), t, Z])
    coeff, residuals, rank, s = np.linalg.lstsq(X, Rk, rcond=None)
    a, b, c = coeff
    fitted = X @ coeff
    resid = Rk - fitted
    RSS = np.sum(resid**2)
    TSS = np.sum((Rk - np.mean(Rk))**2)
    R2 = 1 - RSS/TSS

    Xred = np.column_stack([np.ones_like(t), t])
    coeff_red, _, _, _ = np.linalg.lstsq(Xred, Rk, rcond=None)
    fitted_red = Xred @ coeff_red
    RSS_red = np.sum((Rk - fitted_red)**2)
    df_num = 1
    df_den = len(Rk) - 3
    F = ((RSS_red - RSS) / df_num) / (RSS / df_den)
    p_value_F = 1 - spstats.f.cdf(F, df_num, df_den)

    return a, b, c, R2, resid, F, p_value_F

def log_fit_local_residuals(df_sub):
    Lk = df_sub['interval_right'].values
    Rk = df_sub['Rk'].values
    _, _, _, _, residuals = log_fit(Lk, Rk)
    return residuals