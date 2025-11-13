# cosmology.py

import numpy as np
from scipy.integrate import quad
import constants as c

# Forward model implementation. This module does the following:
# Implementing D_L(z) (numerical). Using scipy.integrate.quad
# Implementing D_L(z) (pen approximation)
# Computes the distance modulus (mu)

def eta_func(a, omega_m):
    """
    Computes the polynomial approximation term eta(a, Omega_m) used in the 
    Pen (1999) luminosity distance fitting formula for a flat universe.

    Parameters
    ----------
    a : float
        The scale factor ratio term (either 1 or 1/(1+z)).
    omega_m : float
        The matter density parameter (Omega_m).

    Returns
    --------
    float
        The value of the eta function.
    """
    s_cubed = (1 - omega_m) / omega_m
    s = s_cubed **(1/3)

    eta = 2 * np.sqrt(s_cubed + 1) * (
        (1 / a**4) 
        - (0.1540 * (s / a**3))
        + (0.4304 * (s**2 / a**2)) 
        + (0.19097 * (s_cubed / a))
        + (0.066941 * s**4)
    )**(-1/8)
    return eta

def integrand(z_prime, omega_m):
    """
    The integrand, 1/E(z'), required for the numerical calculation of the 
    luminosity distance in a flat universe.

    Parameters
    ----------
    z_prime : float
        The variable of integration (redshift).
    omega_m : float
        The matter density parameter (Omega_m).

    Returns
    --------
    float
        The value of 1/E(z').
    """
    E_z_prime = np.sqrt(omega_m * (1.0 + z_prime)**3 + (1.0 - omega_m))
    return 1.0 / E_z_prime

def luminosity_distance(z, omega_m, h, method='pen'):
    """
    Computes the luminosity distance (D_L) in Mpc for a flat universe, 
    using either the Pen approximation or numerical integration.
    
    Parameters
    ----------
    z : float or array
        The redshift of the supernova.
    omega_m : float
        The matter density parameter (Omega_m).
    h : float
        The normalized Hubble constant (H_0 = 100h km/s/Mpc).
    method : str
        The calculation method: 'pen' (approximation, faster) or 
        'numerical' (integration, slower). Defaults to 'pen'.

    Returns
    --------
    float or array
        The luminosity distance D_L in Mpc.
    """
    H_0 = 100 * h # km/s/Mpc

    if method == 'pen':
        return (c.CSOL_KM_S / H_0) * (1.0 + z) * (eta_func(1.0, omega_m) - eta_func(1.0 / (1.0 + z), omega_m))

    else:
        factor = (c.CSOL_KM_S * (1.0 + z)) / H_0
        integral_result, error = quad(integrand, 0.0, z, args=(omega_m,))
        return factor * integral_result

def distance_modulus(z, omega_m, h, method='pen'):
    """
    Computes the distance modulus (mu) in magnitudes from the luminosity 
    distance D_L.

    Parameters
    ----------
    z : float or array
        The redshift of the supernova.
    omega_m : float
        The matter density parameter (Omega_m).
    h : float
        The normalized Hubble constant (h).
    method : str
        The calculation method passed to `luminosity_distance`. Defaults to 'pen'.

    Returns
    --------
    float or array
        The distance modulus mu in magnitudes (mag).
    """
    d = luminosity_distance(z, omega_m, h, method=method) # DL in Mpc
    mu = 25 + (5 * np.log10(d))
    return mu

