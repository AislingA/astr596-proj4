# likelihood.py

# Defines the probability distribution for the parameters given the
#supernova data. This module does the following:
# Implements log_prior()
# Implements log_likelihood()
# Implements log_posterior()

import numpy as np
import cosmology as c
from scipy.linalg import cho_solve
from load_data import load_jla_data

z_data, mu_obs, C_factor = load_jla_data()

def log_prior(theta):
    """
    Computes the log of the prior probability distribution
    A flat (uniform) prior is used within the physically motivated bounds.
    
    Parameters
    ----------
    theta : tuple or numpy.ndarray
        The parameter vector (Omega_m, h).

    Returns
    --------
    float
        0 if parameters are within bounds (log of a constant), 
        or -infinity if outside bounds.
    """
    omega_m, h = theta

    omega_m_min, omega_m_max = 0, 1.2
    h_min, h_max = 0.5, 0.9

    if (omega_m_min < omega_m < omega_m_max) and (h_min < h < h_max):
        # log of a constant (flat prior) is a constant, which can be 0 for relative probs
        return 0
    else:
        return -np.inf

def log_likelihood(theta, z_data=z_data, mu_obs=mu_obs, C_factor=C_factor):
    """
    Computes the log-likelihood function for the 
    Type Ia supernova data assuming Gaussian errors with full covariance.

    The inverse covariance operation is handled 
    by the Cholesky solver.

    Parameters
    ----------
    theta : tuple or numpy.ndarray
        The parameter vector (Omega_m, h).
    z_data : numpy.ndarray
        Observed redshifts.
    mu_obs : numpy.ndarray
        Observed distance.
    C_factor : tuple
        The pre-computed Cholesky factorization of the covariance matrix

    Returns
    --------
    float
        The log-likelihood value.
    """
    omega_m, h = theta

    # calc mu_theory
    mu_theory = c.distance_modulus(z_data, omega_m, h, method='pen')

    # compute residual vector r
    r = mu_obs - mu_theory

    C_inv_r = cho_solve(C_factor, r)

    chi_squared = np.dot(r, C_inv_r)

    log_L = -0.5 * chi_squared

    return log_L

def log_posterior(theta):
    """
    Computes the log of the posterior probability distribution,
    by summing the log-likelihood and log-prior.

    Parameters
    ----------
    theta : tuple or numpy.ndarray
        The parameter vector (Omega_m, h)

    Returns
    --------
    float
        The log-posterior value. Returns -infinity if the prior is violated.
    """
    log_p = log_prior(theta)

    if log_p == -np.inf:
        return -np.inf
    
    log_L = log_likelihood(theta)

    return log_L + log_p