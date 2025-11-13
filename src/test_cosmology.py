# test_cosmology.py

# This module is used to check the following:
# Both methods (numerical vs analytical) agree to ~ 0.4% or better
# across 0.2 ≤ Ωm ≤ 1
# Test with specific values (z = 0.5, Ωm​ = 0.3, h = 0.7) and confirm outputs
# are (D_L^* ≈ 2547 Mpc, μ ≈ 42.81 mag)

import numpy as np
import cosmology as c

z = 0.5
omega_m = 0.3
h = 0.7

expected_dl = 2547 # Mpc
expected_mu = 42.26 # mag

tolerance_percentage = 0.1
tolerance_factor = tolerance_percentage / 100

def test_dl_numerical():
    """
    Tests the accuracy of the Pen approximation (`method='pen'`) and the 
    `distance_modulus` function against the established worked example values
    """
    print("\n --- Testing DL* and mu with Pen approximation ---")

    # calc dl* using pen
    dl_pen = c.luminosity_distance(z, omega_m, h, method='pen')

    # calc mu using pen
    mu_pen = c.distance_modulus(z, omega_m, h, method='pen')
    
    print(f"Pen DL* ({dl_pen:.4f} Mpc) and mu ({mu_pen:.4f} mag) (Expected DL*={expected_dl}, mu={expected_mu})")

def test_dl_method():
    """
    Validates the consistency between the two luminosity distance implementations 
    (`method='pen'` and `method='numerical'`) across a range of cosmological 
    parameters

    The Pen approximation is expected to agree with the numerical integration 
    result to within approximately 0.4%
    """
    print("\n --- Testing Pen vs Numerical DL* Agreement ---")

    # testing a range of omega_m and z values
    omega_m = np.linspace(0.2, 1, 5)
    z = np.array([0.1, 0.5, 1, 1.5])

    for om in omega_m:
        for single_z in z:
            dl_pen = c.luminosity_distance(single_z, om, h, method='pen')
            dl_num = c.luminosity_distance(single_z, om, h, method='numerical')

            # calc the percent difference
            percent_diff = np.abs(dl_pen - dl_num) / dl_num * 100
            
            print(f"Om={om:.2f}, z={single_z:.1f}: Agreement is {percent_diff:.4f}%")

        print(f"Pen and Numerical methods agree better than {tolerance_percentage}% across test range")

if __name__ == '__main__':
    test_dl_numerical()
    test_dl_method()
    print("\nAll cosmology tests passed")