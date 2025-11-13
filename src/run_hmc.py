# run_hcmc.py

import numpy as np
import os
from load_data import load_jla_data
from hmc import hamiltonian_monte_carlo
import time

N_steps = 50000
burn_in_fraction = 0.2
N_chains = 4
init_theta = (0.3, 0.7)
output_dir = "../outputs/chains/"

epsilon = 0.009
L = 11

def run_hmc():
    """
    Executes the full Hamiltonian Monte Carlo (HMC) production run.

    Parameters
    ----------
    (None)

    Returns
    --------
    (None) - Executes the sampling process and saves outputs
    """
    load_jla_data()
    os.makedirs(output_dir, exist_ok=True)
    
    # compiling first
    _ = hamiltonian_monte_carlo(init_theta, epsilon, L, 100, burn_in=0)

    acceptance_rates = []

    start_time = time.time()

    for i in range(N_chains):
        chain, rate, _ = hamiltonian_monte_carlo(
            init_theta, epsilon, L, N_steps, burn_in=0
        )
        acceptance_rates.append(rate)
        fname = os.path.join(output_dir, f"hmc_chain_{i+1}")
        np.save(fname, chain)
        print(f"Acceptance Rate: {rate:.2f}. Chain saved to {fname}")

    end_time = time.time()
    total_sampling_time = end_time - start_time
    print(f"HMC Runtime: {total_sampling_time}")

if __name__ == '__main__':
    run_hmc()