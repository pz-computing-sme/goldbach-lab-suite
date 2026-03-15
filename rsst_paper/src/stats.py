import numpy as np
import pandas as pd

def singular_series_moments(df, limits):
    """
    Compute mean, E[(S-1)^2], and E[S^2] of the singular series S.
    """
    results = []
    df_even = df[df['n'] % 2 == 0].copy()
    for L in limits:
        subset = df_even[df_even['n'] <= L]
        S = subset['S'].values
        mean_S = np.mean(S)
        var_S1 = np.mean((S - 1)**2)   # E[(S-1)^2]
        mean_S2 = np.mean(S**2)        # E[S^2]
        results.append({
            'L': L,
            'mean_S': mean_S,
            'E_Sminus1_sq': var_S1,
            'E_S_sq': mean_S2
        })
    return results

def tilde_R_statistics(df, limits):
    """
    Compute mean and variance of tilde{R}(n) = 2G(n) / (S(n) * n/ln^2 n)
    and the quantity Q(L) = L * var / log^2 L.
    """
    results = []
    df_even = df[df['n'] % 2 == 0].copy()
    # Compute tildeR for each n
    logn = np.log(df_even['n'].values)
    denominator = df_even['S'].values * df_even['n'].values / logn**2
    tildeR = 2 * df_even['G'].values / denominator
    df_even['tildeR'] = tildeR

    for L in limits:
        subset = df_even[df_even['n'] <= L]
        mean_t = subset['tildeR'].mean()
        var_t = subset['tildeR'].var(ddof=0)  # population variance
        Q = L * var_t / (np.log(L)**2)
        results.append({
            'L': L,
            'mean_tildeR': mean_t,
            'var_tildeR': var_t,
            'Q': Q
        })
    return results

def compute_subintervals(df, n_intervals=30):
    """
    Divide the range [1e6, 1e7] into n_intervals equal subintervals.
    For each interval, compute:
        - mean G
        - mean n/ln^2 n (pointwise)
        - estimated singular series = 2*mean_G / mean_n_ln2
        - R(I_k) = mean_G / (L_k/ln^2 L_k)   (using right endpoint)
    Returns a DataFrame with one row per interval.
    """
    # Filter data between 1e6 and 1e7
    df_range = df[(df['n'] >= 1e6) & (df['n'] <= 1e7) & (df['n'] % 2 == 0)].copy()
    df_range = df_range.sort_values('n')
    n_min = 1e6
    n_max = 1e7
    step = (n_max - n_min) / n_intervals

    intervals = []
    for i in range(n_intervals):
        left = n_min + i * step
        right = n_min + (i+1) * step
        mask = (df_range['n'] >= left) & (df_range['n'] < right)
        sub = df_range[mask]
        if len(sub) == 0:
            continue
        mean_G = sub['G'].mean()
        logn = np.log(sub['n'].values)
        term = sub['n'].values / logn**2
        mean_n_ln2 = np.mean(term)
        S_est = 2 * mean_G / mean_n_ln2
        Lk = right
        Rk = mean_G / (Lk / np.log(Lk)**2)
        intervals.append({
            'interval_left': left,
            'interval_right': right,
            'mean_G': mean_G,
            'mean_n_ln2': mean_n_ln2,
            'S_est': S_est,
            'Rk': Rk
        })
    return pd.DataFrame(intervals)