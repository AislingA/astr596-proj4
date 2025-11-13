# hmc.py

# Implements the Hamiltonian MC Sampler, using the leapfrog integrator 
# from previous project. This module does the following:
# Implements the leapfrog integrator
# Implements hamiltonian_monte_carlo()

import numpy as np
from likelihood import log_posterior

def compute_log_posterior_gradient(theta, h=1e-6):
    """
    Computes the gradient of the log-posterior
    using the central finite difference approximation.
    
    Parameters
    ----------
    theta : numpy.ndarray
        The current parameter vector (Omega_m, h).
    h : float
        The finite difference step size. Defaults to 1e-6.

    Returns
    --------
    numpy.ndarray
        The gradient vector
    """
    n_params = len(theta)
    gradient = np.zeros(n_params)

    # making a copy of theta so i do not rewrite the original
    theta_plus = np.array(theta, dtype=float)
    theta_minus = np.array(theta, dtype=float)

    # computing the partial derivatives for each param
    for i in range(n_params):
        # not starting at ith param
        # instead adding h (small val)
        theta_plus[i] += h
        log_p_plus = log_posterior(theta_plus)
        # resetting theta plus 
        theta_plus[i] -= h

        # now doing the same with theta_minus
        theta_minus[i] -= h
        log_p_minus = log_posterior(theta_minus)
        # resetting theta plus 
        theta_minus[i] += h

        # calc the central finite diff
        gradient[i] = (log_p_plus - log_p_minus) / (2 * h)

    return gradient

def leapfrog_int(theta, p, grad_U_fn, epsilon, L):
    """
    Performs L leapfrog integration steps to simulate the Hamiltonian dynamics 
    of the system in parameter space.
    
    Parameters
    ----------
    theta : numpy.ndarray
        Initial position (parameter vector).
    p : numpy.ndarray
        Initial momentum vector.
    grad_U_fn : callable
        Function that computes the gradient of the potential energy
    epsilon : float
        The step size (time step).
    L : int
        The number of leapfrog steps (trajectory length).

    Returns
    --------
    theta_new : numpy.ndarray
        New position vector after L steps.
    p_new : numpy.ndarray
        New momentum vector after L steps.
    """
    # creating copies to avoid issues later
    theta_new = np.copy(theta)
    p_new = np.copy(p)

    # half step momentum
    grad_U_init = grad_U_fn(theta_new)
    p_new -= (epsilon / 2) * grad_U_init

    # full step L
    for _ in range(L - 1):
        theta_new += epsilon * p_new

        # full step momentum
        grad_U = grad_U_fn(theta_new)
        p_new -= epsilon * grad_U

    # final full step pos
    theta_new += epsilon * p_new

    # half step momentum
    grad_U_final = grad_U_fn(theta_new)
    p_new -= (epsilon / 2) * grad_U_final

    return theta_new, p_new



def hamiltonian_monte_carlo(init_theta, epsilon, L, n_steps, burn_in=0):
    """
    Implements the core Hamiltonian Monte Carlo (HMC) sampler algorithm.
    
    The algorithm involves: sampling momentum (p), simulating dynamics via 
    `leapfrog_int`, and accepting/rejecting the new state based on the change 
    in the total Hamiltonian (Delta H).

    Parameters
    ----------
    init_theta : tuple or array
        The starting point for the chain (Omega_m, h).
    epsilon : float
        Leapfrog step size.
    L : int
        Number of leapfrog steps (trajectory length).
    n_steps : int
        Total number of HMC steps to run.
    burn_in : int
        Number of initial steps to discard. Defaults to 0.

    Returns
    --------
    chain : numpy.ndarray
        The HMC samples after burn-in.
    acceptance_rate : float
        The ratio of accepted to proposed steps (target 60-80%).
    energy_diffs : numpy.ndarray
        The energy change (Delta H) at each step, used for convergence diagnostics.
    """
    n_params = len(init_theta)
    chain = np.zeros((n_steps, n_params))
    energy_diffs = np.zeros(n_steps)

    theta_current = np.array(init_theta)
    log_p_current = log_posterior(theta_current)
    n_accepted = 0
    rng = np.random.default_rng()

    # util functions 
    def potential_energy(theta):
        return -log_posterior(theta)
    
    def kinetic_energy(p):
        return 0.5 * np.dot(p, p)
    
    def hamiltonian(theta, p):
        return potential_energy(theta) + kinetic_energy(p)
    
    def grad_U_fn(theta):
        grad_log_p = compute_log_posterior_gradient(theta)
        return - grad_log_p
    
    for i in range(n_steps):
        # sampling momentum (p) from a gaussian
        p_current = rng.normal(size=n_params)
        # computing init hamiltonian
        H_current = hamiltonian(theta_current, p_current)
        # sim w leapfrog
        theta_proposal, p_proposal = leapfrog_int(theta_current, p_current, grad_U_fn, epsilon, L)
        # calc the proposed hamiltonian
        H_proposal = hamiltonian(theta_proposal, p_proposal)
        # calc the acceptance ratio (based on energy change)
        Delta_H = H_proposal - H_current
        # acceptance prob
        alpha = np.min([1, np.exp(-Delta_H)])
        energy_diffs[i] = Delta_H

        # accepting/rejecting
        if rng.random() < alpha:
            theta_current = theta_proposal
            log_p_current = log_posterior(theta_current)
            n_accepted += 1

        chain[i, :] = theta_current

    acceptance_rate = n_accepted / n_steps

    return chain[burn_in:], acceptance_rate, energy_diffs