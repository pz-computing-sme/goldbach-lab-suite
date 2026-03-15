"""
All figure generation functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MultipleLocator
from statsmodels.tsa.stattools import acf
import seaborn as sns
from . import fits

plt.rcParams['font.size'] = 12

def figure_1_global_fit(L_vals, R_vals, output_path):
    """
    Figure 1 (original) / Figure 2 (nova ordem): Global R(L) vs 1/log L.
    """
    L_vals = np.asarray(L_vals)
    R_vals = np.asarray(R_vals)
    t = 1.0 / np.log(L_vals)
    y = R_vals

    a, b, R2, _, _ = fits.log_fit(L_vals, y)

    t_matrix = t.reshape(-1, 1)
    y_fixed = y - 0.5
    b_fixed, _, _, _ = np.linalg.lstsq(t_matrix, y_fixed, rcond=None)
    b_fixed = b_fixed[0]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(t, y, color='blue', label='Data')

    t_fit = np.linspace(min(t), max(t), 100)
    y_fit = a + b * t_fit
    ax.plot(t_fit, y_fit, 'r-', label=f'Free fit: R = {a:.4f} + {b:.4f}/log L')

    y_fixed_fit = 0.5 + b_fixed * t_fit
    ax.plot(t_fit, y_fixed_fit, 'g--', label=f'Fixed intercept: R = 0.5 + {b_fixed:.4f}/log L')

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
    Figure 2 (original) / Figure 3 (nova ordem): Local R(I_k) vs 1/log L_k.
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
    Figure 3 (original) / Figure 4 (nova ordem): Comparação entre ajustes.
    """
    Lk = df_sub['interval_right'].values
    Rk = df_sub['Rk'].values

    a_log, b_log, _, _, _ = fits.log_fit(Lk, Rk)
    a_zt, b_zt, c_zt, R2_zt, _, F, p = fits.fit_with_zeros(df_sub, zeros)

    x_plot = 1.0 / np.log(Lk)
    idx = np.argsort(x_plot)
    x_sorted = x_plot[idx]

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
    Figure 4 (original) / Figure 5 (nova ordem): Autocorrelation of residuals.
    Uses stem plot to match article style.
    """
    residuals = np.asarray(residuals)
    residuals = residuals[~np.isnan(residuals)]
    n = len(residuals)
    print(f"Number of residuals for autocorrelation: {n} (expected 500)")

    # Center
    residuals = residuals - np.mean(residuals)

    lag_max = 50
    acf_vals = acf(residuals, nlags=lag_max, fft=True)
    lags = np.arange(len(acf_vals))  # includes lag 0

    # 95% confidence band
    conf = 1.96 / np.sqrt(n)
    print(f"Confidence band: {conf:.4f}")

    fig, ax = plt.subplots(figsize=(10,5))
    # Use stem plot for discrete lags
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
    Figure 5 (original) / Figure 6 (nova ordem): Q(L) = L * var(tildeR) / log^2 L.
    Plots Q(L) vs L with log scale on x-axis, using seaborn styling.
    """
    # Extract L and Q values from tilde_stats
    L_vals = [d['L'] for d in tilde_stats]
    Q_vals = [d['Q'] for d in tilde_stats]

    # Create DataFrame for seaborn
    df_plot = pd.DataFrame({'L': L_vals, 'Q': Q_vals})

    # Set seaborn style and context
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)

    plt.figure(figsize=(8, 6))
    sns.lineplot(x='L', y='Q', data=df_plot, marker='o', linewidth=2, markersize=10, color='blue')
    plt.xscale('log')
    plt.xlabel('L', fontsize=14)
    plt.ylabel(r'$Q(L) = \frac{L \cdot \operatorname{Var}(\tilde{R})}{\log^2 L}$', fontsize=14)
    plt.title('Mean-square error in the pre-asymptotic regime (Theorem 6.1)', fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def figure_6_hist3d(df, output_path):
    """
    Figure 6 (original) / Figure 1 (nova ordem): 3D histogram.
    High-definition version matching article style.
    """
    df_even = df[df['n'] % 2 == 0].copy()
    # Use columns 'n', 'G', 'S' (S is the singular series)
    n_vals = df_even['n'].values
    G_vals = df_even['G'].values
    S_vals = df_even['S'].values

    n_max_val = n_vals.max()
    g_max_val = G_vals.max()

    # Binning
    n_bins = 30
    g_bins = 30

    # Create bins
    n_edges = np.linspace(0, n_max_val / 1e6, n_bins + 1)
    g_edges = np.linspace(0, g_max_val, g_bins + 1)

    # Compute 2D histogram with weights = S
    H, xedges, yedges = np.histogram2d(
        n_vals / 1e6, G_vals, bins=[n_edges, g_edges], weights=S_vals)
    counts, _, _ = np.histogram2d(
        n_vals / 1e6, G_vals, bins=[n_edges, g_edges])

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_S = np.divide(H, counts)
        avg_S[counts == 0] = 0

    # Prepare bar positions
    n_centers = (n_edges[:-1] + n_edges[1:]) / 2
    g_centers = (g_edges[:-1] + g_edges[1:]) / 2
    n_mesh, g_mesh = np.meshgrid(n_centers, g_centers, indexing='ij')

    dx = (n_edges[1] - n_edges[0]) * 0.85
    dy = (g_edges[1] - g_edges[0]) * 0.85

    xpos = n_mesh.flatten()
    ypos = g_mesh.flatten()
    zpos = np.zeros_like(xpos)
    dz = avg_S.flatten()

    # Remove NaN bins
    mask = ~np.isnan(dz)
    xpos = xpos[mask]
    ypos = ypos[mask]
    zpos = zpos[mask]
    dz = dz[mask]

    # Styling
    norm = Normalize(vmin=dz.min() - 0.8, vmax=dz.max())
    colors = cm.inferno(norm(dz))

    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
             color=colors, alpha=1.0, shade=True,
             edgecolor='black', linewidth=0.4)

    # Axis formatting
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

    ax.view_init(elev=25, azim=-65)

    ax.set_title('3D Histogram: Average Singular Series by n and G(n)\n($4 \\leq n \\leq 10^7$)',
                 fontsize=18, pad=30, fontweight='bold')

    # Colorbar
    sm = cm.ScalarMappable(cmap='inferno', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label(r'Average $\mathfrak{S}(n)$', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()