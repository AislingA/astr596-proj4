# run_mcmc.py

import numpy as np
import os
from load_data import load_jla_data
from mcmc import metropolis_hastings
import time

N_steps = 50000
burn_in_fraction = 0.2
N_chains = 4
init_theta = (0.3, 0.7)
output_dir = "../outputs/chains/"

def run_mcmc():
    """
    Executes the full MCMC production run.

    Parameters
    ----------
    (None)

    Returns
    --------
    (None) - Executes the sampling process and saves outputs
    """
    load_jla_data()
    os.makedirs(output_dir, exist_ok=True)
    chains = []
    acceptance_rates = []
    # tuning
    sigma_om = 0.05
    sigma_h = 0.02
    proposal_covariance = np.diag([sigma_om**2, sigma_h**2])

    start_time = time.time()

    for i in range(N_chains):
        chain, rate = metropolis_hastings(
            init_theta, proposal_covariance, N_steps, burn_in=0
        )
        chains.append(chain)
        acceptance_rates.append(rate)
        fname = os.path.join(output_dir, f"mcmc_chain_{i+1}")
        np.save(fname, chain)
        print(f"Acceptance Rate: {rate:.2f}. Chain saved to {fname}")

    end_time = time.time()
    total_sampling_time = end_time - start_time
    print(f"MCMC Runtime: {total_sampling_time}")

if __name__ == '__main__':
    run_mcmc()