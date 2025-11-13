# plotting.py

# this module makes the following plots:
# trace plots
# autocorreltation plots
# corner plots
# data vs model plots
# diagnostic plots
# efficiency comparison plots
# energy change plot

import matplotlib.pyplot as plt
import numpy as np
import diagnostics as d
import corner
import cosmology as c

def trace_plot(chains, param_names, fname='trace_plot.png', title='MCMC Trace Plot'):
    """
    Generates trace plots, showing parameter values versus iteration number for multiple chains.
    This plot is used to visually diagnose burn-in, mixing, and stationarity of the chains.

    Parameters
    ----------
    chains : list of numpy.ndarray
        List of parameter chains (N_samples x N_params).
    param_names : list of str
        Names of the parameters.
    fname : str, optional
        Output filename. Defaults to 'trace_plot.png'.
    title : str, optional
        Plot title.
    """
    n_params = chains[0].shape[1]
    n_chains = len(chains)

    fig, axes = plt.subplots(n_params, 1, figsize=(10, 4 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for j in range(n_params):
        for i in range(n_chains):
            ax = axes[j]
            chain = chains[i]
            iterations = np.arange(chain.shape[0])

            # overlaying all chains w diff colors
            ax.plot(iterations, chain[:, j], lw=0.5, alpha=0.8, label=f'Chain {i+1}')

        ax.set_ylabel(f"${param_names[j]}$", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel("Iteration", fontsize=14)
    fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fname)
    plt.close(fig)

def autocorrelation_plot(chains, parameter_names, max_lag=50, filename="autocorrelation_plot.png", title="Autocorrelation Function"):
    """
    Generates plots of the Autocorrelation Function versus lag k for each parameter.
    The decay rate is directly related to the chain's statistical efficiency.

    Parameters
    ----------
    chains : list of numpy.ndarray
        List of parameter chains (N_samples x N_params).
    parameter_names : list of str
        Names of the parameters.
    max_lag : int, optional
        Maximum lag to display. Defaults to 50.
    filename : str, optional
        Output filename.
    title : str, optional
        Plot title.
    """
    n_params = chains[0].shape[1]
    
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 4 * n_params), sharex=True)
    
    if n_params == 1:
        axes = [axes]
    
    # flatten all chains to one array
    pooled_chain = np.concatenate(chains, axis=0)

    for j in range(n_params):
        ax = axes[j]
        # computing ACF for the param samples
        acf = d.compute_autocorrelation(pooled_chain[:, j], max_lag=max_lag)
        lags = np.arange(len(acf))
        ax.stem(lags, acf, markerfmt=".")
        
        # here, I add in 95% confidence bounds
        N_total = len(pooled_chain[:, j])
        # also adding in a dashed line to help visualize
        ax.axhline(0.1, color='red', linestyle='--', alpha=0.7, lw=1)
        ax.axhline(0.0, color='gray', linestyle='-', lw=1)
        ax.set_ylabel(f'ACF ($\\rho_k$) for ${parameter_names[j]}$', fontsize=14)
        ax.set_ylim(-0.2, 1.0)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    axes[-1].set_xlabel(f"Lag $k$ (steps)", fontsize=14)
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    plt.close(fig)

def diagnostic_plot(chains, parameter_names, burn_in=0, filename="gelman_rubin_plot.png", title="Gelman-Rubin Convergence (R_hat)"):
    """
    Generates a bar chart displaying the final Gelman-Rubin statistic for each parameter.
    Convergence is typically indicated when R < 1.1

    Parameters
    ----------
    chains : list of numpy.ndarray
        List of parameter chains (N_samples x N_params).
    parameter_names : list of str
        Names of the parameters.
    burn_in : int, optional
        Number of steps to discard before calculating R. Defaults to 0.
    filename : str, optional
        Output filename.
    title : str, optional
        Plot title.
    """
    # computing R_hat on post-burn-in samples
    post_burn_in_chains = [c[burn_in:] for c in chains]
    R_hat_values = d.gelman_rubin(post_burn_in_chains)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(parameter_names, R_hat_values, color=['#1f77b4', '#ff7f0e'])
    # plotting the convergence threshold
    ax.axhline(1.1, color='red', linestyle='--', label='Convergence Threshold (1.1)')
    
    # adding R_hat values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', 
                ha='center', va='bottom', fontsize=12)

    ax.set_ylabel('Gelman-Rubin $\hat{R}$', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_ylim(min(0.95, np.min(R_hat_values)-0.05), max(1.15, np.max(R_hat_values)+0.05))
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

def energy_hist(energy_diffs, epsilon, L, filename="energy_hist_plot.png"):
    """
    Generates a histogram of the energy change between the proposed 
    and current HMC states. 

    Parameters
    ----------
    energy_diffs : numpy.ndarray
        Array of Delta H values recorded during the HMC run.
    epsilon : float
        The HMC step size used.
    L : int
        The HMC trajectory length used.
    filename : str, optional
        Output filename.
    """
    plt.figure(figsize=(8,5))
    plt.hist(energy_diffs, bins=50, density=True, alpha=0.7, color='b')
    plt.title(f'HMC Energy Conservation (epsilon={epsilon}, L={L})')
    plt.xlabel("Energy CHange ($\Delta H = H_{prop} - H_{curr}$)")
    plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.6)

    # putting a line for where dist should be peaked
    plt.axvline(np.mean(energy_diffs), color='r', linestyle='-', label=f"Mean $\Delta H$: {np.mean(energy_diffs):.3e}")
    plt.legend()
    plt.savefig(filename)

def corner_plot(chains_mcmc, chains_hmc, param_names, fname="corner_plot.png", title="Posterior Distribution Comparison"):
    """
    Generates a corner plot showing the posterior 
    distributions for the cosmological parameters. This function compares the 
    MCMC and HMC results by overplotting the contours.

    Parameters
    ----------
    chains_mcmc : list of numpy.ndarray
        List of post-burn-in MCMC chains.
    chains_hmc : list of numpy.ndarray
        List of post-burn-in HMC chains.
    param_names : list of str
        Names of the parameters.
    fname : str, optional
        Output filename.
    title : str, optional
        Plot title.
    """
    # pooling samples
    pooled_mcmc = np.concatenate(chains_mcmc, axis=0)
    pooled_hmc = np.concatenate(chains_hmc, axis=0)
    param_labels = [f'$\\Omega_m$', f'$h$']

    fig = corner.corner(
        pooled_mcmc,
        labels=param_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={'fontsize': 12},
        color='b',
        truth_color='r',
        hist_kwargs={"density": True},
        fig=plt.figure(figsize=(8,8))
    )

    corner.corner(
        pooled_hmc,
        fig=fig,
        color='orange',
        hist_kwargs={"density": True},
        plot_datapoints=False,
        plot_density=False,
        plot_contours=True,
        no_fill_contours=True,
        contours=1,
        alpha=0.6,
        fill_contours=False
    )

    fig.suptitle(title, fontsize=16)
    fig.savefig(fname)
    plt.close(fig)

def data_v_model_plot(mu_obs, z_data, chains_hmc, param_names, fname="data_v_model.png", title="Supernova Data vs Posterior Model Predictions"):
    """
    Generates a plot comparing the observed JLA data against 
    a visualization of the model uncertainty. This is done by overplotting 
    model curves generated from a random subset of posterior samples.

    Parameters
    ----------
    mu_obs : numpy.ndarray
        Observed distance moduli.
    z_data : numpy.ndarray
        Observed redshifts.
    chains_hmc : list of numpy.ndarray
        List of post-burn-in HMC chains (used for model visualization).
    param_names : list of str
        Names of the parameters.
    fname : str, optional
        Output filename.
    title : str, optional
        Plot title.
    """
    pooled_samples = np.concatenate(chains_hmc, axis=0)
    N_samples_plot = 100

    inds = np.random.randint(0, len(pooled_samples), N_samples_plot)
    random_theta = pooled_samples[inds]
    z_plot = np.linspace(np.min(z_data), np.max(z_data), 100)
    fig, ax = plt.subplots(figsize=(10,6))

    for theta in random_theta:
        om, h = theta[0], theta[1]
        mu_theory = c.distance_modulus(z_plot, om, h, method='pen')
        ax.plot(z_plot, mu_theory, color='b')

    ax.scatter(z_data, mu_obs, marker='o', color='r', zorder=5, label="JLA Data Points")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Distance Modulus")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)

def eff_comparison_plot(eff_dict, fname="eff_comparison.png", title="ESS per second"):
    """
    Generates a bar chart comparing the Effective Samples per Second (ESS/sec) 
    for different samplers (MCMC vs. HMC)

    Parameters
    ----------
    eff_dict : dict
        Dictionary containing sampler names as keys and their calculated ESS/sec 
        efficiency as values.
    fname : str, optional
        Output filename.
    title : str, optional
        Plot title.
    """
    samplers = list(eff_dict.keys())
    ess_per_sec = list(eff_dict.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(samplers, ess_per_sec, color=['r', 'b'])
    ax.set_ylabel("Effective Samples per second (ESS/sec)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)