# constants.py

# This file contains a collection of physical and astronomical constants
# defined in the CGS (centimeter-gram-second) unit system.

# Solar constants in CGS
LSUN = 3.826e33 # Luminosity of the sun in ergs/s
RSUN = 6.955e10 # Radius of the sun in cm
MSUN = 1.989e33 # Mass of the sun in g
TSUN = 5777 # Temperature of the sun in K

# Other constants in CGS
G = 6.674e-8 # Gravitational constant in cm^3/g/s^2
SIGMA_SB = 5.67e-5 # Stefan-Boltzmann constant in erg/s/cm^2/K^4
WIEN_B = 0.289 # Wien's displacement constant in cm * K
CSOL = 3e10 # Speed of light in cm/s
H = 6.626e-27 # planck's constant in erg*s
K_B = 1.381e-16 # boltzmann constant in erg/K
PC = 3.086e18 # parsec in cm
M_H = 1.673e-24 # proton mass in grams

# non CGS
CSOL_KM_S = 3e5 # speed of light in km/s

# Time constants in CGS
YEAR = 3.156e7 # Julian year in s
GYR = 1e9 * YEAR # Gyr in s
MYR = 1e6 * YEAR # Myr in s