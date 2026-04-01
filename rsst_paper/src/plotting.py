"""
All figure generation functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MultipleLocator
from statsmodels.tsa.stattools import acf
from . import fits

plt.rcParams['font.size'] = 12
sns.set_theme(style="whitegrid", palette="muted")

def figure_1_global_fit(L_vals, R_vals, output_path):
    """
    Figure 2: Global R(L) vs 1/log L, with free and fixed-intercept fits.
    Exactly matches the style of the example script.
    """
    L_vals = np.asarray(L_vals)
    R_vals = np.asarray(R_vals)
    x = 1.0 / np.log(L_vals)
    y = R_vals

    # Free linear regression (as in example)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    R2_free = r_value**2

    # Fit with intercept fixed at 0.5 (as in example)
    b_fixed = np.mean((y - 0.5) / x)
    residuals_fixed = y - (0.5 + b_fixed * x)
    ss_res_fixed = np.sum(residuals_fixed**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2_fixed = 1 - ss_res_fixed / ss_tot

    # Plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=x, y=y, color='blue', s=80, edgecolor='w', linewidth=0.5, label='Global data', zorder=5)

    x_fit = np.linspace(min(x), max(x), 200)
    y_fit_free = intercept + slope * x_fit
    plt.plot(x_fit, y_fit_free, 'r-', linewidth=2.5,
             label=f'Free fit: $R = {intercept:.3f} + {slope:.3f}/\\ln L$')

    y_fit_fixed = 0.5 + b_fixed * x_fit
    plt.plot(x_fit, y_fit_fixed, 'g--', linewidth=2,
             label=f'Fixed intercept: $R = 0.5 + {b_fixed:.3f}/\\ln L$')

    plt.xlabel(r'$1/\ln L$', fontsize=14)
    plt.ylabel(r'$R(L)$', fontsize=14)
    plt.title('Global fit of cumulative data', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def figure_2_local_fit(df_sub, output_path):
    """
    Figure 3: Local R(I_k) vs 1/log L_k.
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
    Figure 4: Comparison of logarithmic fit and fit including zero term.
    Uses seaborn style with smooth curves and metrics box.
    """
    Lk = df_sub['interval_right'].values
    Rk = df_sub['Rk'].values

    # Fit logarithmic model (free intercept)
    a_log, b_log, r2_log, _, _ = fits.log_fit(Lk, Rk)

    # Fit full model (with zeros)
    a_full, b_full, c_full, r2_full, resid_full, F, p_value, _ = fits.fit_with_zeros(df_sub, zeros)

    x_plot = 1.0 / np.log(Lk)
    # Sort for smooth curves
    idx = np.argsort(x_plot)
    x_sorted = x_plot[idx]
    L_sorted = Lk[idx]

    # Smooth prediction lines
    x_smooth = np.linspace(min(x_plot), max(x_plot), 200)
    L_smooth = np.exp(1 / x_smooth)
    Z_smooth = np.array([fits.zero_term(L, zeros) for L in L_smooth])

    y_log_smooth = a_log + b_log * x_smooth
    y_full_smooth = a_full + b_full * x_smooth + c_full * Z_smooth

    # Metrics for text box
    resid_log = Rk - (a_log + b_log * x_plot)
    rmse_log = np.sqrt(np.mean(resid_log**2))
    rmse_full = np.sqrt(np.mean(resid_full**2))
    rmse_reduction = (rmse_log - rmse_full) / rmse_log * 100

    plt.figure(figsize=(10, 6))
    # Observed data
    sns.scatterplot(x=x_plot, y=Rk, color='black', s=80, label='Observed', zorder=5, edgecolor='white', linewidth=1)

    # Logarithmic model
    plt.plot(x_smooth, y_log_smooth, 'b-', linewidth=2.5,
             label=f'Log model: $R = {a_log:.3f} + {b_log:.3f}/\\ln L$')

    # Full model
    plt.plot(x_smooth, y_full_smooth, 'r-', linewidth=2.5,
             label=f'Full model: $R = {a_full:.3f} + {b_full:.3f}/\\ln L + {c_full:.3f}Z_{{\\mathrm{{full}}}}$')

    plt.xlabel(r'$1/\ln L$', fontsize=14)
    plt.ylabel(r'$R(I_k)$', fontsize=14)
    plt.title('Comparison of models with full zero term', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Text box with metrics
    textstr = f'$R^2$ (log): {r2_log:.4f}\n$R^2$ (full): {r2_full:.4f}\nRMSE reduction: {rmse_reduction:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.65)
    plt.text(0.04, 0.75, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def figure_4_autocorr(residuals, output_path):
    """
    Figure 5: Autocorrelation of residuals (stem plot).
    """
    residuals = np.asarray(residuals)
    residuals = residuals[~np.isnan(residuals)]
    n = len(residuals)
    print(f"Number of residuals for autocorrelation: {n} (expected 500)")

    residuals = residuals - np.mean(residuals)

    lag_max = 50
    acf_vals = acf(residuals, nlags=lag_max, fft=True)
    lags = np.arange(len(acf_vals))  # includes lag 0

    conf = 1.96 / np.sqrt(n)
    print(f"Confidence band: {conf:.4f}")

    fig, ax = plt.subplots(figsize=(10,5))
    markerline, stemlines, baseline = ax.stem(
        lags[1:], acf_vals[1:], basefmt=" ", markerfmt='o', linefmt='steelblue')
    plt.setp(markerline, markersize=4, color='steelblue')
    plt.setp(stemlines, linewidth=1, color='steelblue')

    ax.axhline(y=conf, linestyle='--', color='gray', label='95% confidence')
    ax.axhline(y=-conf, linestyle='--', color='gray')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'Autocorrelation of residuals ({n} intervals)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, lag_max+1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def figure_5_Q_plot(tilde_stats, output_path):
    """
    Figure 6: Q(L) = L * var(tildeR) / log^2 L.
    """
    L_vals = [d['L'] for d in tilde_stats]
    Q_vals = [d['Q'] for d in tilde_stats]

    df_plot = pd.DataFrame({'L': L_vals, 'Q': Q_vals})

    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)

    plt.figure(figsize=(8, 6))
    sns.lineplot(x='L', y='Q', data=df_plot, marker='o', linewidth=2, markersize=10, color='blue')
    plt.xscale('log')
    plt.xlabel('L', fontsize=14)
    plt.ylabel(r'$Q(L) = \frac{L \cdot \operatorname{Var}(\tilde{R})}{\log^2 L}$', fontsize=14)
    plt.title('Mean-square error in the pre-asymptotic regime', fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def figure_6_hist3d(df, output_path):
    """
    Figure 1: 3D histogram exactly matching the vivid script.
    """
    # 1. Filter even numbers only
    df_even = df[df['n'] % 2 == 0].copy()

    # 2. Binning Logic (30 bins each)
    n_bins, g_bins = 30, 30
    n_max_val = df_even['n'].max()
    g_max_val = df_even['G'].max()

    # Create bins using pd.cut (same as script)
    df_even['n_bin'] = pd.cut(df_even['n'], bins=n_bins, labels=False)
    df_even['g_bin'] = pd.cut(df_even['G'], bins=g_bins, labels=False)

    # Pivot table for average S (singular series)
    pivot = df_even.pivot_table(values='S', index='g_bin', columns='n_bin', aggfunc='mean')

    # 3. Coordinate preparation (same as script)
    n_edges = np.linspace(0, n_max_val / 1e6, n_bins + 1)
    g_edges = np.linspace(0, g_max_val, g_bins + 1)

    n_centers = (n_edges[:-1] + n_edges[1:]) / 2
    g_centers = (g_edges[:-1] + g_edges[1:]) / 2
    n_mesh, g_mesh = np.meshgrid(n_centers, g_centers)

    dx = (n_edges[1] - n_edges[0]) * 0.85
    dy = (g_edges[1] - g_edges[0]) * 0.85

    xpos = n_mesh.flatten()
    ypos = g_mesh.flatten()
    zpos = np.zeros_like(xpos)
    dz = pivot.values.flatten()

    # Remove NaN bins (no data)
    mask = ~np.isnan(dz)
    xpos, ypos, zpos, dz = xpos[mask], ypos[mask], zpos[mask], dz[mask]

    if len(dz) == 0:
        print("Warning: No data for 3D histogram.")
        return

    # 4. Styling and Plotting
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111, projection='3d')

    # Color mapping: inferno with vmin shifted (same as script)
    norm = Normalize(vmin=dz.min() - 0.8, vmax=dz.max())
    colors = cm.inferno(norm(dz))

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
             color=colors, alpha=1.0, shade=True,
             edgecolor='black', linewidth=0.4)

    # 5. Axis Formatting (exactly as script)
    def x_format(x, pos):
        return "0" if x == 0 else f"{int(x)}M"

    def y_format(y, pos):
        return "0" if y == 0 else f"{int(y/1000)}k"

    ax.xaxis.set_major_formatter(FuncFormatter(x_format))
    ax.yaxis.set_major_formatter(FuncFormatter(y_format))
    ax.yaxis.set_major_locator(MultipleLocator(20000))

    ax.tick_params(axis='x', which='major', pad=8)
    ax.tick_params(axis='y', which='major', pad=15)

    ax.set_xlabel('n (Millions)', fontsize=14, labelpad=20, fontweight='bold')
    ax.set_ylabel('G(n)', fontsize=14, labelpad=30, fontweight='bold')
    ax.set_zlabel(r'Average $\mathfrak{S}(n)$', fontsize=14, labelpad=15, fontweight='bold')

    ax.set_xlim(0, n_edges.max())
    ax.set_ylim(0, g_edges.max())
    ax.set_zlim(0, dz.max() + 0.5)

    # 6. Perspective and Title
    ax.view_init(elev=25, azim=-65)

    ax.set_title('3D Histogram: Average Singular Series by n and G(n)\n($4 \\leq n \\leq 10^7$)',
                 fontsize=18, pad=30, fontweight='bold')

    # Colorbar
    sm = cm.ScalarMappable(cmap='inferno', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label(r'Average $\mathfrak{S}(n)$', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()