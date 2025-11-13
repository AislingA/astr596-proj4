# test_hcmc.py

import numpy as np
import matplotlib.pyplot as plt
from load_data import load_jla_data
from hmc import hamiltonian_monte_carlo
from plotting import energy_hist

if __name__ == '__main__':
    print("--- HMC Check ---")
    
    # load the data
    z_data, mu_data, C_factor = load_jla_data()

    # hmc params
    init_theta = (0.3, 0.7)
    epsilon = 0.009
    L = 11
    n_steps = 1000
    burn_in = 100

    # running hmc
    chain, rate, energy_diffs = hamiltonian_monte_carlo(
        init_theta, epsilon, L, n_steps, burn_in=0
    )
    print(f"Target Acceptance Rate: 60-80%. HMC Acceptance Rate: {rate:.2f}")

    # diagnostics
    # energy conservation plot
    energy_hist(energy_diffs, epsilon, L)

    if rate < 0.6:
        print("\nAcceptance rate is too low, decrease epsilon.")
    elif rate > 0.8:
        print("\nAcceptance rate is too high, increase epsilon.")