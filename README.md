# Project 4: Universal Inference Engine for Cosmology

This repository contains the code for Project 4, which involves building and comparing general-purpose MCMC and Hamiltonian Monte Carlo (HMC) samplers. These engines are applied to the JLA Type Ia supernova dataset to infer the cosmological parameters of a flat universe: the matter density ($\Omega_m$) and the normalized Hubble constant ($h$).

##  Scientific Goal

To measure the composition of the cosmos by determining the posterior distribution for the cosmological parameters ($\Omega_m, h$), which controls the expansion history of the universe.

## Installation and Setup

### Prerequisites

You need Python 3.8+ and the following external libraries.

```bash
pip install -r requirements.txt
```
### Data Setup

The project relies on two data files:

    jla_mub.txt: Contains the 31 binned supernova data points.

    jla_mub_covmatrix.txt: Contains the full 31×31 covariance matrix.

## How to Run
Execute the following scripts, which will save the chains and timing data to ./outputs/chains/.

### Run MCMC production (saves chains and time)
```bash
python run_mcmc.py
```
# Run HMC production (saves chains and time)
```bash
python run_hmc.py
```

### Run Analysis and Plotting
Execute the analysis script to load the saved chains, compute ESS/sec, and generate all final plots.
```bash
python analyze_results.py
```