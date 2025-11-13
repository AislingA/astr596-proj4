# load_data.py

import numpy as np
from scipy.linalg import cho_factor

def load_jla_data():
    """
    Loads the Type Ia supernova data (redshift and distance modulus) 
    and the full covariance matrix
    
    The covariance matrix is prepared for efficient likelihood 
    calculation by computing its Cholesky factorization.

    Parameters
    ----------
    (None)

    Returns
    --------
    z_data : numpy.ndarray
        Array of observed redshifts z_i.
    mu_data : numpy.ndarray
        Array of observed distance moduli mu_i.
    C_factor : tuple
        The pre-computed Cholesky factorization of the covariance matrix, 
        used in the log-likelihood calculation.
    """
    data = np.loadtxt('../data/jla_mub.txt')
    z_data = data[:, 0]
    mu_data = data[:, 1]

    c_elements = np.loadtxt('../data/jla_mub_covmatrix.txt')

    N = len(z_data)

    # verfiy sizes
    assert len(mu_data) == N, f"mu_data has {len(mu_data)} elements, expected {N}"
    assert c_elements.size == N * N, f"Covariance matrix has {c_elements.size} elements, expected {N*N}"

    # reshaping the 1d array of elements into a N x N matrix
    C = c_elements.reshape((N, N))

    # computing c factor using cholesky
    C_factor = cho_factor(C)

    assert C.shape == (N ,N), f"Covariance matrix C shape is {C.shape}, expected ({N}, {N})"

    print(f"\n --- JLA Data Load ---")
    print(f"Data loaded successfully. N={N}")
    print(f"Covariance matrix C computed and Cholesky factored. C_factor computed.")

    return z_data, mu_data, C_factor