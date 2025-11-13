# diagnostics.py

# Tools to assess sampler performance, usable for MCMC and HMC. This module
# does the following:
# Implements compute_autocorrelation()
# Implements effective_sample_size()
# Implements gelman_rubin()

import numpy as np

def compute_autocorrelation(chain, max_lag=None):
    """
    Computes the autocorrelation function (ACF), rho_k, for a 1D chain of samples.
    
    Parameters
    ----------
    chain : numpy.ndarray
        A 1D array of samples for a single parameter.
    max_lag : int, optional
        Maximum lag K to compute the ACF up to

    Returns
    --------
    numpy.ndarray
        The autocorrelation values rho_k for lags k=0 to max_lag.
    """
    N = len(chain)
    if N < 2:
        return np.array([1])
    
    # subtract the mean
    mu = np.mean(chain)
    centered_chain = chain - mu

    # computing the denom (variance)
    variance = np.var(chain)
    if variance == 0:
        return np.zeros(N // 2 if max_lag is None else max_lag + 1)
    
    # computing the num (auto covariance)
    acov = np.correlate(centered_chain, centered_chain, mode='full')
    acov = acov[N - 1:]
    acf = acov / (N * variance)

    # getting the limit for result
    limit = N // 2 if max_lag is None else max_lag + 1
    
    return acf[:limit]

def effective_sample_size(chain):
    """
    Computes the Effective Sample Size (ESS) for a 1D chain of samples. 
    The ESS quantifies the number of independent samples contained in the chain.
    
    Parameters
    ----------
    chain : numpy.ndarray
        A 1D array of samples for a single parameter.

    Returns
    --------
    float
        The calculated Effective Sample Size, capped at the chain length N.
    """
    N = len(chain)
    if N < 2:
        return float(N)

    # computing for the full chain
    acf = compute_autocorrelation(chain, max_lag=N - 1)
    
    tau = 1.0 
    sum_rho_k = 0.0
    for k in range(1, len(acf)):
        sum_rho_k += acf[k]
        # if the sum is negative than we rollback and break
        if (1 + 2 * sum_rho_k) < 0: 
             sum_rho_k -= acf[k]
             break

    tau = 1.0 + 2.0 * sum_rho_k
    ess = N / tau
    
    return min(N, ess) # since ESS cannot be bigger than the chain length

def gelman_rubin(chains):
    """
    Computes the Gelman-Rubin split-R statistic (Potential Scale Reduction Factor) 
    by comparing within-chain variance (W) to between-chain variance (B).
    Convergence is typically achieved when R < 1.1

    Parameters
    ----------
    chains : list of numpy.ndarray
        A list of numpy arrays, where each array is a chain (N_samples x N_params). 

    Returns
    --------
    numpy.ndarray
        The R value for each parameter.
    """
    # num of chains
    M = len(chains)
    if M < 2:
        # if number is less than 2, r_hat cannot be computed: 
        return np.ones(chains[0].shape[1]) * np.nan
    
    # splitting each chain in half
    split_chains = []
    for chain in chains:
        N = chain.shape[0]
        split_point = N // 2
        split_chains.append(chain[:split_point])
        split_chains.append(chain[split_point:])
    M_split = 2 * M
    N_split = split_chains[0].shape[0]
    
    # variance of each split chain
    W_m = np.array([np.var(c, axis=0, ddof=1) for c in split_chains])
    W = np.mean(W_m, axis=0) # averaging w
    
    # the mean of each split chain
    theta_bar_m = np.array([np.mean(c, axis=0) for c in split_chains])
    # the mean of all split means
    theta_bar = np.mean(theta_bar_m, axis=0)
    
    B = (N_split / (M_split - 1)) * np.sum((theta_bar_m - theta_bar)**2, axis=0)
    # estimating the posterior variance
    V_hat = ((N_split - 1) / N_split) * W + (1.0 / N_split) * B
    # the potential scale reduction factor
    R_hat = np.sqrt(V_hat / W)
    return R_hat