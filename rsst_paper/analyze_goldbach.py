#!/usr/bin/env python3
"""
Main driver script for the Goldbach analysis.
Generates all tables (except Table 1) and figures from the paper,
using the pointwise dataset up to 10^7.
"""

import os
import pandas as pd
from src import data_loader, stats, fits, plotting

def main():
    # -------------------- Configuration --------------------
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    out_dir = os.path.join(base_dir, 'output')
    tables_dir = os.path.join(out_dir, 'tables')
    figures_dir = os.path.join(out_dir, 'figures')

    for d in [tables_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)

    # -------------------- Load data --------------------
    print("Loading Goldbach data (up to 10^7)...")
    df_goldbach = data_loader.load_goldbach_data(
        os.path.join(data_dir, 'goldbach_full.csv')
    )

    print("Loading zeta zeros...")
    zeros = data_loader.load_zeros(
        os.path.join(data_dir, 'zeta_zeros_100000.txt'),
        n_zeros=1000
    )

    # -------------------- Singular series moments (Table 3) --------------------
    print("Computing singular series moments (Table 3)...")
    # Use only even n
    df_even = df_goldbach[df_goldbach['n'] % 2 == 0].copy()
    limits = [1e4, 1e5, 1e6, 1e7]   # 10^8 not available in CSV
    ss_moments = stats.singular_series_moments(df_even, limits)
    pd.DataFrame(ss_moments).to_csv(
        os.path.join(tables_dir, 'table3.csv'), index=False
    )

    # -------------------- Tilde{R} statistics (Section 7.5) --------------------
    print("Computing tilde{R} statistics...")
    tilde_stats = stats.tilde_R_statistics(df_even, limits)
    pd.DataFrame(tilde_stats).to_csv(
        os.path.join(tables_dir, 'tilde_R.csv'), index=False
    )

    # -------------------- Subinterval analysis (30 intervals) --------------------
    print("Computing 30 subintervals...")
    df_sub_30 = stats.compute_subintervals(df_goldbach, n_intervals=30)
    df_sub_30.to_csv(os.path.join(tables_dir, 'R_subintervals.csv'), index=False)

    # -------------------- Figures --------------------
    print("Generating figures...")

    # Figure 1: Global fit – uses data from Table 1 (already published)
    # We'll load the values from the article manually (or from a small CSV)
    # For reproducibility, we include a tiny CSV with the five points.
    global_points = pd.read_csv(os.path.join(data_dir, 'global_R_points.csv'))
    plotting.figure_1_global_fit(
        global_points['L'].values,
        global_points['R'].values,
        os.path.join(figures_dir, 'figure_1.pdf')
    )

    # Figure 2: Local fit (30 subintervals)
    plotting.figure_2_local_fit(
        df_sub_30,
        os.path.join(figures_dir, 'figure_2.pdf')
    )

    # Figure 3: Fit with zero term
    plotting.figure_3_zeros_fit(
        df_sub_30,
        zeros,
        os.path.join(figures_dir, 'figure_3.pdf')
    )

    # Figure 4: Autocorrelation (500 subintervals)
    print("Computing 500 subintervals for residual analysis...")
    df_sub_500 = stats.compute_subintervals(df_goldbach, n_intervals=500)
    resid_500 = fits.log_fit_local_residuals(df_sub_500)
    plotting.figure_4_autocorr(
        resid_500,
        os.path.join(figures_dir, 'figure_4.pdf')
    )

    # Figure 5: Q(L) plot
    plotting.figure_5_Q_plot(
        tilde_stats,
        os.path.join(figures_dir, 'figure_5.pdf')
    )

    # Figure 6: 3D histogram
    plotting.figure_6_hist3d(
        df_goldbach,
        os.path.join(figures_dir, 'figure_6.pdf')
    )

    print("All done! Tables and figures saved in output/.")

if __name__ == '__main__':
    main()