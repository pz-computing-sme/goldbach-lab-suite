"""
All figure generation functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from . import fits

# Set global matplotlib style (optional)
plt.rcParams['font.size'] = 12

def figure_1_global_fit(global_stats, output_path):
    """
    Figure 1: Global R(L) vs 1/log L with linear fit.
    global_stats is list of dicts from stats.global_statistics.
    """
    L_vals = [d['L'] for d in global_stats]
    R_vals = [d['R'] for d in global_stats]

    x = 1.0 / np.log(L_vals)
    y = R_vals

    # Fit
    a, b, R2, _, _ = fits.log_fit(L_vals, y)

    # Create plot
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(x, y, color='blue', label='Data')
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = a + b * x_fit
    ax.plot(x_fit, y_fit, 'r-', label=f'Fit: R = {a:.4f} + {b:.4f}/log L')
    # Optional: fit with intercept fixed at 0.5
    y_fixed = 0.5 + b * x_fit
    ax.plot(x_fit, y_fixed, 'g--', label='Intercept = 0.5')

    ax.set_xlabel(r'$1/\log L$')
    ax.set_ylabel(r'$R(L)$')
    ax.set_title('Global fit of $R(L)$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def figure_2_local_fit(df_sub, output_path):
    """
    Figure 2: Local R(I_k) vs 1/log L_k with linear fit.
    df_sub has columns interval_right and Rk.
    """
    Lk = df_sub['interval_right'].values
    Rk = df_sub['Rk'].values
    x = 1.0 / np.log(Lk)
    y = Rk

    a, b, R2, _, _ = fits.log_fit(Lk, y)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(x, y, color='blue', label='Data')
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = a + b * x_fit
    ax.plot(x_fit, y_fit, 'r-', label=f'Fit: R = {a:.4f} + {b:.4f}/log L')
    ax.set_xlabel(r'$1/\log L_k$')
    ax.set_ylabel(r'$R(I_k)$')
    ax.set_title('Local fit for 30 subintervals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def figure_3_zeros_fit(df_sub, zeros, output_path):
    """
    Figure 3: Compare pure log fit vs fit with zero term.
    """
    Lk = df_sub['interval_right'].values
    Rk = df_sub['Rk'].values
    x = Lk  # for plotting we will use 1/log scale as x-axis

    # Pure log fit
    a_log, b_log, _, _, _ = fits.log_fit(Lk, Rk)
    # Zero-term fit
    a_zt, b_zt, c_zt, R2_zt, _, F, p = fits.fit_with_zeros(df_sub, zeros)

    # Prepare x axis in 1/log scale for plotting
    x_plot = 1.0 / np.log(Lk)
    # Sort for smooth lines
    idx = np.argsort(x_plot)
    x_sorted = x_plot[idx]
    Rk_sorted = Rk[idx]

    # Predictions
    y_log = a_log + b_log * x_sorted
    y_zt = a_zt + b_zt * x_sorted + c_zt * np.array([fits.zero_term(L, zeros) for L in Lk[idx]])

    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(x_plot, Rk, color='blue', s=20, label='Data')
    ax.plot(x_sorted, y_log, 'b-', label='Log fit', linewidth=2)
    ax.plot(x_sorted, y_zt, 'r-', label='Log + zeros fit', linewidth=2)
    ax.set_xlabel(r'$1/\log L_k$')
    ax.set_ylabel(r'$R(I_k)$')
    ax.set_title('Comparison of fits (30 subintervals)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def figure_4_autocorr(residuals, output_path):
    """
    Figure 4: Autocorrelation function of residuals.
    residuals: 1D array of residuals from 500 subintervals.
    """
    from statsmodels.tsa.stattools import acf
    lag_max = 40
    acf_vals = acf(residuals, nlags=lag_max, fft=False)
    lags = np.arange(len(acf_vals))

    # Confidence bands (approx 95%)
    conf = 1.96 / np.sqrt(len(residuals))

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(lags, acf_vals, width=0.3, color='steelblue', edgecolor='black')
    ax.axhline(y=conf, linestyle='--', color='gray', label='95% CI')
    ax.axhline(y=-conf, linestyle='--', color='gray')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Residual autocorrelation (500 subintervals)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def figure_5_Q_plot(tilde_stats, output_path):
    """
    Figure 5: Q(L) = L * var(tildeR) / log^2 L vs L.
    tilde_stats is list of dicts from stats.tilde_R_statistics.
    """
    L_vals = [d['L'] for d in tilde_stats]
    Q_vals = [d['Q'] for d in tilde_stats]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(L_vals, Q_vals, 'o-', color='darkgreen', markersize=8)
    ax.set_xlabel('$L$')
    ax.set_ylabel('$Q(L)$')
    ax.set_title(r'$Q(L) = \frac{L \cdot \operatorname{Var}(\tilde{R})}{\log^2 L}$')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def figure_6_hist3d(df, output_path):
    """
    Figure 6: 3D histogram of average singular series as function of n and G.
    """
    # Prepare data: use log10 bins for n? We'll use linear bins for simplicity.
    # n in millions: n_mill = n / 1e6
    df_even = df[df['n'] % 2 == 0].copy()
    df_even['n_mill'] = df_even['n'] / 1e6
    G = df_even['G'].values
    S = df_even['S'].values
    n_mill = df_even['n_mill'].values

    # Create 2D bins
    n_bins = 20
    G_bins = 20
    H, xedges, yedges = np.histogram2d(n_mill, G, bins=[n_bins, G_bins], weights=S)
    # counts for averaging
    counts, _, _ = np.histogram2d(n_mill, G, bins=[n_bins, G_bins])
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_S = np.divide(H, counts)
        avg_S[counts == 0] = 0  # or np.nan

    # Prepare coordinates for bar plot
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.5*(xedges[1]-xedges[0]),
                              yedges[:-1] + 0.5*(yedges[1]-yedges[0]), indexing='ij')
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    dx = (xedges[1] - xedges[0]) * np.ones_like(xpos)
    dy = (yedges[1] - yedges[0]) * np.ones_like(xpos)
    dz = avg_S.ravel()

    # Mask empty bins
    mask = counts.ravel() > 0
    xpos = xpos[mask]
    ypos = ypos[mask]
    zpos = zpos[mask]
    dx = dx[mask]
    dy = dy[mask]
    dz = dz[mask]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, cmap='viridis')
    ax.set_xlabel('$n$ (millions)')
    ax.set_ylabel('$G(n)$')
    ax.set_zlabel('Average $\mathfrak{S}(n)$')
    ax.set_title('3D histogram: average singular series vs $n$ and $G(n)$')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()