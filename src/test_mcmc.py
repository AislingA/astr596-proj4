# test_mcmc.py

import numpy as np
from load_data import load_jla_data
from mcmc import metropolis_hastings
from plotting import trace_plot, autocorrelation_plot, diagnostic_plot
import diagnostics as d

if __name__ == '__main__':
    # loading the data
    z_data, mu_data, C_factor = load_jla_data()

    # tuning
    sigma_om = 0.05
    sigma_h = 0.02
    proposal_covariance = np.diag([sigma_om**2, sigma_h**2])

    # mcmc params
    init_theta = (0.3, 0.7)
    n_steps = 5000
    burn_in = 1000
    n_chains = 4

    param_names = ['Omega_m', 'h']

    chains = []
    acceptance_rates = []

    print(f"Running {n_chains} MCMC chains")
    for i in range(n_chains):
        chain, rate = metropolis_hastings(init_theta, proposal_covariance, n_steps, burn_in)
        chains.append(chain)
        acceptance_rates.append(rate)
        print(f"Chain {i+1} Acceptance Rate: {rate:.2f}")

    # diagnostics
    post_burnin_chains = [c[burn_in:] for c in chains]
    print("\n--- Diagnostics ---")
    R_hat = d.gelman_rubin(post_burnin_chains)
    print(f"Gelman-Rubin (R_hat): {R_hat}")
    print(f"ESS for Omega_m: {d.effective_sample_size(post_burnin_chains[0][:, 0]):.0f}")
    print(f"ESS for h: {d.effective_sample_size(post_burnin_chains[0][:, 1]):.0f}")

    # plots
    trace_plot(chains, param_names)
    autocorrelation_plot(post_burnin_chains, param_names, max_lag=50)
    diagnostic_plot(chains, param_names, burn_in)