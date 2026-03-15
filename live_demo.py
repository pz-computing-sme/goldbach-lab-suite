import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time
from numba import njit, prange
import math

# ========================================
# 🧮 GOLDBACH ENGINE (OPTIMIZED WITH NUMBA)
# ========================================

def fast_sieve(limit):
    """Efficient NumPy implementation of the Sieve of Eratosthenes."""
    primes = np.ones(limit + 1, dtype=bool)
    primes[0:2] = False
    for p in range(2, int(limit**0.5) + 1):
        if primes[p]:
            primes[p * p : limit + 1 : p] = False
    return np.where(primes)[0]

@njit(parallel=True)
def count_partitions_chunk(start_n, end_n, primes, is_prime):
    """
    Count Goldbach partitions for even numbers from start_n to end_n (inclusive, step 2).
    Returns arrays of n, partitions, density.
    """
    max_size = (end_n - start_n) // 2 + 1
    n_vals = np.zeros(max_size, dtype=np.int64)
    part_vals = np.zeros(max_size, dtype=np.int64)
    dens_vals = np.zeros(max_size, dtype=np.float64)

    idx = 0
    for n in range(start_n, end_n + 1, 2):
        count = 0
        for i in range(len(primes)):
            p = primes[i]
            if p > n // 2:
                break
            if is_prime[n - p]:
                count += 1
        n_vals[idx] = n
        part_vals[idx] = count
        dens_vals[idx] = count / n
        idx += 1
    return n_vals[:idx], part_vals[:idx], dens_vals[:idx]

def calculate_goldbach_data_optimized(limit, num_workers=4):
    """
    Optimized calculation using Numba and chunking.
    """
    primes_arr = fast_sieve(limit)
    # Create boolean array for fast primality test
    max_n = limit
    is_prime_bool = np.zeros(max_n + 1, dtype=np.bool_)
    is_prime_bool[primes_arr] = True

    # Split the even numbers into chunks for parallel processing
    evens = np.arange(4, limit + 1, 2)
    chunk_size = len(evens) // num_workers + 1
    chunks = [(evens[i], evens[min(i + chunk_size - 1, len(evens)-1)]) for i in range(0, len(evens), chunk_size)]

    # Progress bar setup
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_chunks = len(chunks)

    results = []
    for chunk_idx, (start_n, end_n) in enumerate(chunks):
        status_text.text(f"Processing chunk {chunk_idx+1}/{total_chunks}: n = {start_n} to {end_n}")
        n_vals, part_vals, dens_vals = count_partitions_chunk(start_n, end_n, primes_arr, is_prime_bool)
        for n, p, d in zip(n_vals, part_vals, dens_vals):
            results.append({"n": n, "partitions": p, "density": d})
        progress_bar.progress((chunk_idx + 1) / total_chunks)

    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results), len(primes_arr)

# ========================================
# 📊 FUNCTION TO COMPUTE CUMULATIVE INTERVAL STATISTICS
# ========================================

def compute_cumulative_stats(df, max_val):
    """
    For each power of 10 limit (10^4, 10^5, ..., up to max_val),
    compute statistics over the interval [4, limit].
    Returns a DataFrame with the results.
    """
    limits = []
    k = 4
    while 10**k <= max_val:
        limits.append(10**k)
        k += 1
    if max_val not in limits:
        limits.append(max_val)

    stats = []
    for limit in limits:
        subset = df[df['n'] <= limit]
        if subset.empty:
            continue
        num_evens = len(subset)
        min_g = subset['partitions'].min()
        max_g = subset['partitions'].max()
        mean_g = subset['partitions'].mean()
        mean_density = subset['density'].mean()

        if limit > 1:
            n_over_ln2 = limit / (np.log(limit) ** 2)
        else:
            n_over_ln2 = 1

        cp_mean = mean_g / n_over_ln2

        stats.append({
            'Interval': f'4 to {limit:,}',
            'Num Evens': num_evens,
            'Min G(n)': int(min_g),
            'Max G(n)': int(max_g),
            'Mean G(n)': round(mean_g, 2),
            'Mean Density (G(n)/n)': round(mean_density, 6),
            'C (from mean)': round(c_mean, 4)
        })
    return pd.DataFrame(stats)

# ========================================
# 🖥️ STREAMLIT INTERFACE (ACADEMIC)
# ========================================

st.set_page_config(page_title="Goldbach Heuristic Lab", layout="wide", page_icon="🔬")

st.title("🔬 Goldbach Conjecture: Heuristic & Numerical Lab")
st.markdown(f"""
**Author:** Vitor Pozza  
**Framework:** Reduced Sum Set Test (RSST)  
**Status:** Heuristic Evidence for Mathematics Research & Preprints  
---
""")

with st.expander("📝 METHODOLOGY & HEURISTIC ARGUMENT"):
    st.write("""
    ### 1. Research Objective
    This laboratory explores the **probabilistic abundance** of Goldbach partitions $G(n)$. It provides high‑quality numerical evidence supporting the asymptotic behavior of prime sums.
    
    ### 2. The Scarcity Paradox
    The fundamental observation identifies that while the **Relative Density** $D(n) = G(n)/n$ decays logarithmically, the absolute count of partitions diverges. The RSST framework analyzes the safety margin between the lowest partition count and zero.
    
    ### 3. Expectations vs. Deterministic Certainty
    In rigorous mathematics, a high expectation ($E[G(n)] \sim n/\ln^2 n$) is a strong indicator but does not eliminate the theoretical possibility of a counter‑example. This lab visualizes why such a "Goldbach Hole" becomes statistically impossible at large scales.
    """)

st.sidebar.header("⚙️ Lab Parameters")
st.sidebar.info("Scaling up to 10M is recommended for deep heuristic analysis. Larger values may take significant time.")
max_val = st.sidebar.number_input("Numerical Limit (n)", min_value=100, max_value=100000000, value=10000, step=10000)
num_workers = st.sidebar.slider("Number of parallel workers", min_value=1, max_value=8, value=4, step=1)
run_btn = st.sidebar.button("Run Numerical Analysis")

if run_btn:
    start_time = time.time()
    df, total_primes = calculate_goldbach_data_optimized(max_val, num_workers=num_workers)
    end_time = time.time()

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Range Sampled", f"4 to {max_val:,}")
    m2.metric("Primes in Range", f"{total_primes:,}")
    m3.metric("Min. G(n) Observed", f"{int(df['partitions'].min()):,}")

    # --- PLOT 1: GOLDBACH'S COMET ---
    st.header("🌠 1. Goldbach's Comet")
    st.markdown("Visualizing the **Absolute Count** of partitions. $G(n)$ = Count of Pairs $(p_1,p_2)$ plotted using the Magma color scale to highlight density growth.")
    fig_comet = px.scatter(df, x="n", y="partitions", color="partitions",
                           color_continuous_scale="magma",
                           labels={"partitions": "G(n)", "n": "Even Number (n)"})
    fig_comet.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig_comet, use_container_width=True)

    st.divider()

    # --- PLOT 2: SCARCITY ---
    st.header("📉 2. Relative Partition Density")
    st.markdown("The **Relative Scarcity Curve**: This graph represents $G(n)/n$.")
    fig_density = px.line(df, x="n", y="density",
                          color_discrete_sequence=["#00FF88"],
                          labels={"density": "G(n)/n", "n": "Even Number (n)"})
    fig_density.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig_density, use_container_width=True)

    # --- CUMULATIVE STATISTICS ---
    st.divider()
    st.header("📊 Cumulative Interval Statistics")
    st.latex(r"""
    \begin{aligned}
    \text{Num Evens} &= \#\{n \text{ even} \mid 4 \le n \le L\} \\
    \min G(L) &= \min_{4 \le n \le L} G(n) \\
    \max G(L) &= \max_{4 \le n \le L} G(n) \\
    \overline{G}(L) &= \frac{1}{\text{Num Evens}} \sum_{4 \le n \le L} G(n) \\
    \overline{D}(L) &= \frac{1}{\text{Num Evens}} \sum_{4 \le n \le L} \frac{G(n)}{n} \\
    C(L) &= \frac{\overline{G}(L)}{L / \ln^2 L}
    \end{aligned}
    """)

    stats_df = compute_cumulative_stats(df, max_val)
    if not stats_df.empty:
        st.dataframe(stats_df, use_container_width=True)
        last_row = stats_df.iloc[-1]
        st.info(f"**Constant estimate (from mean):** {last_row['C (from mean)']} "
                f"for interval {last_row['Interval']}")
    else:
        st.warning("Not enough data to compute interval statistics.")

    st.divider()
    st.success(f"Computation completed in {end_time - start_time:.2f} seconds.")
    st.header("📊 Numerical Interpretation")
    st.warning("""
    **Heuristic Rigor:** While the divergence of $G(n)$ is evident, this study serves as a numerical verification of the conjecture's robustness rather than a deterministic proof.
    """)
    st.markdown(f"""
    1. **Empirical Robustness:** Within the range of **{max_val:,}**, the minimum number of prime partitions found was **{int(df['partitions'].min()):,}**.
    2. **Trend Correlation:** The absolute count (Comet) diverges positively, while the relative density (Curve) stabilizes, suggesting the absence of Goldbach holes in the tested limits.
    """)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export Heuristic Data (CSV)", csv, f"goldbach_heuristic_{max_val}.csv")

else:
    st.info("Adjust parameters in the sidebar and click 'Run Numerical Analysis' to start.")

st.divider()
st.caption("© 2026 Vitor Pozza Research | Heuristic Study on Number Theory")