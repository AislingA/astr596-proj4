# analyze_results.py

import numpy as np
import glob
import os
from diagnostics import effective_sample_size, gelman_rubin
from hmc import hamiltonian_monte_carlo
import plotting as p

N_chains = 4
burn_in_fraction = 0.2
sampler_types = ["MCMC", "HMC"]
param_names = ["Omega_m", "h"]

timing_results = {
    'MCMC': 49.57349109649658,
    'HMC': 850.0628502368927
} # seconds

def load_and_pool_chains(sampler_type):
    """
    Loads all independent chains for a given sampler type, discards the 
    burn-in portion, and returns the list of post-burn-in chains.

    Parameters
    -----------
    sampler_type : str
        The type of sampler chains to load ('MCMC' or 'HMC').
    
    Returns
    --------
    list of numpy.ndarray
        A list where each element is a single chain containing only the 
        post-burn-in samples (N_valid_samples x N_parameters).
    """
    f_pattern = f"../outputs/chains/{sampler_type.lower()}_chain_*.npy"
    chain_files = sorted(glob.glob(f_pattern))

    all_post_burnin_samples = []
    for fname in chain_files:
        full_chain = np.load(fname)
        burnin_steps = int(full_chain.shape[0] * burn_in_fraction)
        post_burnin_samples = full_chain[burnin_steps:, :]
        all_post_burnin_samples.append(post_burnin_samples)

    return all_post_burnin_samples

def analyze_samples(chains, sampler):
    """
    Calculates and prints the full statistical summary for the pooled samples 
    for a given sampler.

    Parameters
    -----------
    chains : list of numpy.ndarray
        List of post-burn-in chains for a single sampler type.
    sampler : str
        The name of the sampler ('MCMC' or 'HMC')
    
    Returns
    --------
    float
        The calculated efficiency metric: Effective Samples per Second (ESS/sec).
    """
    pooled_samples = np.concatenate(chains, axis=0)
    om_samples = pooled_samples[:, 0]
    h_samples = pooled_samples[:, 1]

    mean_om = np.mean(om_samples)
    std_om = np.std(om_samples)

    mean_h = np.mean(h_samples)
    std_h = np.std(h_samples)

    ess_om = effective_sample_size(om_samples)
    ess_h = effective_sample_size(h_samples)

    min_ess = min(ess_om, ess_h)
    ess_per_sec = min_ess / timing_results[sampler]
    print(f"Gelman-Rubin: {gelman_rubin(chains)}")

    percentiles_om = np.percentile(om_samples, [16, 50, 84])
    percentiles_h = np.percentile(h_samples, [16, 50, 84])

    print("\nCredible Intervals (16th | 50th | 84th Percentile):")
    print(f"Omega_m: {percentiles_om[0]:.4f} | {percentiles_om[1]:.4f} | {percentiles_om[2]:.4f}")
    print(f"h:       {percentiles_h[0]:.4f} | {percentiles_h[1]:.4f} | {percentiles_h[2]:.4f}")
    return ess_per_sec


if __name__ == '__main__':
    mcmc_chains = load_and_pool_chains("MCMC")
    ess_mcmc = analyze_samples(mcmc_chains, "MCMC")
    hmc_chains = load_and_pool_chains("HMC")
    ess_hmc = analyze_samples(hmc_chains, "HMC")
    
    from load_data import load_jla_data
    z_data, mu_obs, C_factor = load_jla_data()

    init_theta = (0.3, 0.7)
    epsilon = 0.009
    L = 11
    n_steps = 1000
    burn_in = 100
    _, _, energy_diffs = hamiltonian_monte_carlo(
        init_theta, epsilon, L, n_steps, burn_in=0
    )

    eff_results = {
        "MCMC": ess_mcmc,
        "HMC": ess_hmc
    }

    # plots
    print("\n--- Generating Deliverable Plots ---")
    
    p.trace_plot(mcmc_chains, param_names, fname="../outputs/figures/mcmc_trace.png", title="MCMC Trace Plot (Post Burn-in)")
    p.trace_plot(hmc_chains, param_names, fname="../outputs/figures/hmc_trace.png", title="HMC Trace Plot (Post Burn-in)")
    p.autocorrelation_plot(mcmc_chains, param_names, filename="../outputs/figures/mcmc_acf.png", title="MCMC Autocorrelation Function")
    p.autocorrelation_plot(hmc_chains, param_names, filename="../outputs/figures/hmc_acf.png", title="HMC Autocorrelation Function")
    p.diagnostic_plot(hmc_chains, param_names, burn_in=0, filename="../outputs/figures/hmc_rhat.png", title="Gelman-Rubin Convergence (HMC)")
    p.energy_hist(energy_diffs, epsilon, L, filename="../outputs/figures/hmc_delta_h.png")
    p.corner_plot(mcmc_chains, hmc_chains, param_names, fname="../outputs/figures/corner_comparison.png", title="Ωm-h Posterior Distribution Comparison")
    p.data_v_model_plot(mu_obs, z_data, hmc_chains, param_names, fname="../outputs/figures/data_vs_model.png")
    p.eff_comparison_plot(eff_results, fname="../outputs/figures/ess_per_sec_comparison.png")
    
    print("\nAll plots saved successfully to ../outputs/figures/")