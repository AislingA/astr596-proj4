# run_comparison.py

import time
import numpy as np
import hmc
import jax_hmc
import diagnostics as d
from load_data import load_jla_data

N_samples = 10000
burn_in = 2000
N_chains = 1
init_theta = (0.3, 0.7)

epsilon = 0.009
L = 11
epsilon_jax = 0.009
L_jax = 11

def run_eff_comparison():
    """"""
    print("--- Running Efficiency Comparison for HMC vs JAX HMC ---")

    # loading in data
    load_jla_data()

    # normal hmc
    start_time_norm = time.time()
    chain_norm, rate_norm, _ = hmc.hamiltonian_monte_carlo(
        init_theta, epsilon, L, N_samples, burn_in=0
    )
    end_time_norm = time.time()
    time_norm = end_time_norm - start_time_norm
    # ESS
    post_burnin_chain_norm = chain_norm[burn_in:, :]
    ess_norm_om = d.effective_sample_size(post_burnin_chain_norm[:, 0])
    ess_norm_h = d.effective_sample_size(post_burnin_chain_norm[:, 1])
    ess_norm = min(ess_norm_om, ess_norm_h)
    ess_per_sec_norm = ess_norm / time_norm
    print("\n--- Efficiency Results ---")
    print(f"Total Samples (post burn in): {N_samples - burn_in}")
    print("\nNormal HMC")
    print(f" Clock Time: {time_norm:.2f} s")
    print(f" Acceptance Rate: {rate_norm:.2f}")
    print(f" Min ESS: {ess_norm:.0f}")
    print(f" ESS per second: {ess_per_sec_norm:.2f}")

    # JAX HMC
    start_time_jax = time.time()
    # doing a jax run to compile JIT
    _ = jax_hmc.hamiltonian_monte_carlo_jax(
        init_theta, epsilon_jax, L_jax, N_samples, burn_in=0
    )
    # second jax run for true performance now that jit is compiled
    chain_jax, rate_jax, _ = jax_hmc.hamiltonian_monte_carlo_jax(
        init_theta, epsilon_jax, L_jax, N_samples, burn_in=0
    )
    end_time_jax = time.time()
    time_jax = end_time_jax - start_time_jax
    # ESS
    post_burnin_chain_jax = chain_jax[burn_in:, :]
    ess_jax_om = d.effective_sample_size(post_burnin_chain_jax[:, 0])
    ess_jax_h = d.effective_sample_size(post_burnin_chain_jax[:, 1])
    ess_jax = min(ess_jax_om, ess_jax_h)
    ess_per_sec_jax = ess_jax / time_jax
    print("\nJAX HMC")
    print(f" Clock Time: {time_jax:.2f} s")
    print(f" Acceptance Rate: {rate_jax:.2f}")
    print(f" Min ESS: {ess_jax:.0f}")
    print(f" ESS per second: {ess_per_sec_jax:.2f}")

    print(f"\nJAX speedup (ESS/sec): {ess_per_sec_jax / ess_per_sec_norm:.1f}x")

if __name__ == '__main__':
    run_eff_comparison()