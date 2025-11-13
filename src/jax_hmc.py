#jax_hmc.py

import numpy as np
import jax
import jax_likelihood as jl
import jax.numpy as jnp

@jax.jit
def potential_energy_jax(theta):
    return -jl.log_posterior_jax(theta)

@jax.jit
def kinetic_energy_jax(p):
    return 0.5 * jnp.dot(p, p)

@jax.jit
def hamiltonian_jax(theta, p):
    return potential_energy_jax(theta) + kinetic_energy_jax(p)

@jax.jit
def grad_U_fn_jax(theta):
        grad_log_p = jl.calc_log_posterior_grad_jax(theta)
        return - grad_log_p

def leapfrog_int(theta, p, grad_U_fn_jax, epsilon, L):
    """"""
    # creating copies to avoid issues later
    theta_new = np.copy(theta)
    p_new = np.copy(p)

    # half step momentum
    grad_U_init = grad_U_fn_jax(theta_new)
    p_new -= (epsilon / 2) * grad_U_init

    # full step L
    for _ in range(L - 1):
        theta_new += epsilon * p_new

        # full step momentum
        grad_U = grad_U_fn_jax(theta_new)
        p_new -= epsilon * grad_U

    # final full step pos
    theta_new += epsilon * p_new

    # half step momentum
    grad_U_final = grad_U_fn_jax(theta_new)
    p_new -= (epsilon / 2) * grad_U_final

    return theta_new, p_new



def hamiltonian_monte_carlo_jax(init_theta, epsilon, L, n_steps, burn_in=0):
    n_params = len(init_theta)
    chain = np.zeros((n_steps, n_params))
    energy_diffs = np.zeros(n_steps)

    theta_current = np.array(init_theta)
    H_current = hamiltonian_jax(jnp.array(theta_current), jnp.zeros(n_params))
    log_p_current = -potential_energy_jax(jnp.array(theta_current))
    n_accepted = 0
    rng = np.random.default_rng()
    
    for i in range(n_steps):
        # sampling momentum (p) from a gaussian
        p_current = rng.normal(size=n_params)
        # computing init hamiltonian
        H_current = hamiltonian_jax(theta_current, p_current)
        # sim w leapfrog
        theta_proposal, p_proposal = leapfrog_int(theta_current, p_current, grad_U_fn_jax, epsilon, L)
        # calc the proposed hamiltonian
        H_proposal = hamiltonian_jax(jnp.array(theta_proposal), jnp.array(p_proposal))
        # calc the acceptance ratio (based on energy change)
        Delta_H = H_proposal - H_current
        # acceptance prob
        alpha = np.min([1, np.exp(-Delta_H)])
        energy_diffs[i] = Delta_H

        # accepting/rejecting
        if rng.random() < alpha:
            theta_current = theta_proposal
            log_p_current = -potential_energy_jax(jnp.array(theta_current))
            n_accepted += 1

        chain[i, :] = theta_current

    acceptance_rate = n_accepted / n_steps

    return chain[burn_in:], acceptance_rate, energy_diffs