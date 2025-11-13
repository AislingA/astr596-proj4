# jax_likelihood.py

import jax
import jax.numpy as jnp
from scipy.linalg import cho_solve
import numpy as np
from load_data import load_jla_data
import jax_cosmology as jc

z_data_np, mu_obs_np, C_factor = load_jla_data()
z_data = jnp.array(z_data_np)
mu_obs = jnp.array(mu_obs_np)

@jax.jit
def log_prior_jax(theta):
    """"""
    omega_m, h = theta
    omega_m_min, omega_m_max = 0.0, 1.2
    h_min, h_max = 0.5, 0.9

    in_bounds = (omega_m >=omega_m_min) & (omega_m <= omega_m_max) & (h >= h_min) & (h <= h_max)

    return jnp.where(in_bounds, 0.0, -jnp.inf)

def chi_squared_jax(theta, z_data, mu_obs, C_factor):
    """"""
    # getting mu_theory
    mu_theory = jc.distance_modulus_jax(z_data, theta[0], theta[1])
    # residual vector
    r = mu_obs - mu_theory
    C_inv_r = jnp.array(cho_solve(C_factor, np.array(r)))
    chi_squared = jnp.dot(r, C_inv_r)
    return chi_squared

@jax.jit
def log_posterior_jax(theta):
    """"""
    log_p = log_prior_jax(theta)
    log_L = jnp.where(log_p > -jnp.inf, -0.5 * chi_squared_jax(theta, z_data, mu_obs, C_factor), -jnp.inf)
    return log_L + log_p

grad_log_posterior_jax = jax.grad(log_posterior_jax)

def calc_log_posterior_grad_jax(theta):
    """"""
    theta_jax = jnp.array(theta)
    grad_jax = grad_log_posterior_jax(theta_jax)
    return np.array(grad_jax)