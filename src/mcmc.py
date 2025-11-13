# mcmc.py

# Implements the general random-walk MCMC sampler. This module does
# the following:
# Implements metropolis_hastings()

import numpy as np
import likelihood as l

def metropolis_hastings(init_theta, proposal_covariance, n_steps, burn_in=0):
    """
    Implements the general Metropolis-Hastings random-walk MCMC sampler.
    
    Parameters
    ----------
    init_theta : tuple or numpy.ndarray
        The starting parameter vector for the chain
    proposal_covariance : numpy.ndarray
        The covariance matrix for the Gaussian proposal distribution.
        This must be tuned to achieve a 20-50% acceptance rate.
    n_steps : int
        Total number of MCMC iterations to perform.
    burn_in : int, optional
        The number of initial steps to discard before returning the chain. Defaults to 0.

    Returns
    --------
    chain : numpy.ndarray
        The array of sampled parameter states after discarding the burn-in period.
    acceptance_rate : float
        The ratio of accepted proposals to total proposed steps.
    """
    n_params = len(init_theta)
    chain = np.zeros((n_steps, n_params))

    theta_current = np.array(init_theta)
    log_p_current = l.log_posterior(theta_current)

    n_accepted = 0

    for i in range(n_steps):
        # proposing a new theta
        theta_proposal = np.random.multivariate_normal(theta_current, proposal_covariance)

        # compute the log-posterior for the proposal
        log_p_proposal = l.log_posterior(theta_proposal)

        # compute the acceptance ratio
        log_alpha = log_p_proposal - log_p_current

        alpha = np.min([1, np.exp(log_alpha)])

        # accepting/rejecting
        if np.random.rand() < alpha:
            theta_current = theta_proposal
            log_p_current = log_p_proposal
            n_accepted += 1

        chain[i, :] = theta_current

    acceptance_rate = n_accepted / n_steps

    return chain[burn_in:], acceptance_rate