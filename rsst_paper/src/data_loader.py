"""
Data loading utilities.
"""

import pandas as pd

def load_goldbach_data(path):
    """
    Load the Goldbach dataset from a CSV file.
    Expected columns: n, G(n), S_singular.
    Renames them to n, G, S for consistency.
    """
    df = pd.read_csv(path)
    print("Original columns:", df.columns.tolist())
    
    # Renomear colunas para os nomes padrão
    df = df.rename(columns={
        'G(n)': 'G',
        'S_singular': 'S'
    })
    # Opcional: remover a coluna extra 'S(n)' se não for necessária
    if 'S(n)' in df.columns:
        df = df.drop(columns=['S(n)'])
    
    # Verificar se as colunas esperadas existem
    required = ['n', 'G', 'S']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Coluna '{col}' não encontrada após renomeação. Colunas disponíveis: {df.columns.tolist()}")
    
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