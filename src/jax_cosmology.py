# jax_cosmology.py

import jax.numpy as jnp
from math import pi
import constants as c

def eta_func_jax(a, omega_m):
    """"""
    s_cubed = (1.0 - omega_m) / omega_m
    s = s_cubed**(1.0/3.0)

    eta = 2 * jnp.sqrt(s_cubed + 1.0) * jnp.power(
        jnp.power(a, -4.0)
        - 0.1540 * (s* jnp.power(a, -3.0))
        + 0.4304 * (jnp.power(s, 2.0) * jnp.power(a, -2.0))
        + 0.19097 * (s_cubed * jnp.power(a, -1.0))
        + 0.066941 * jnp.power(s, 4.0),
        -1.0/8.0
    )
    return eta

def distance_modulus_jax(z, omega_m, h):
    """"""
    H_0 = 100.0 * h # km/s/Mpc

    D_L = (c.CSOL_KM_S / H_0) * (1.0 + z) * (eta_func_jax(1.0, omega_m) - eta_func_jax(1.0 / (1.0+z), omega_m))

    mu = 25.0 + 5.0 * jnp.log10(D_L)
    return mu