"""
Data loading utilities.
"""

import pandas as pd

def load_goldbach_data(path):
    """
    Load the Goldbach dataset from a CSV file.

    Expected columns: n, G, S (where S = singular series)
    """
    df = pd.read_csv(path)
    # Optional: rename columns if necessary
    # df.columns = ['n', 'G', 'S']
    return df

def load_zeros(path, n_zeros=1000):
    """
    Load the imaginary parts of the first n_zeros non-trivial zeros of zeta(s).
    File should contain one gamma per line.
    """
    zeros = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_zeros:
                break
            line = line.strip()
            if line:
                zeros.append(float(line))
    return zeros