# Growth Memo - Project 4

**Name:** Aisling Acuna  
**Date:** 11/12/2025 \
**Project:** Building a Universal Inference Engine

---

### Summary
This project involved building a modular Bayesian inference engine to determine the matter density and hubble constant from JLA Type Ia supernova data. There were two algorithms applied, an MCMC random walk sampler and a HMC gradient-based sampler. The analysis from this project provided parameter estimates consisten with the range for the flat $\lambda$ CDM model.

---

### Technical Skills Developed

- Gradient-Based Sampler: Used the central finite difference method to calculate the log posterior in the HMC sampler.
- Convergence: Applied effective sample size (ESS) and the Gelman-Rubin statistic to validate the convergence of the chains.

---

### Key Challenges & Solutions (or Smooth Sailing)
I had challenges within my extension, where I was trying to apply the jax.gradient. 

**The Problem (or The Approach):**
I wanted to try to apply JAX to use automatic differentation for the HMC gradient. However, I kept facing errors when testing out the JAX code.

**What This Taught Me:**
If I applied more time to the extension, I would have been able to figure out the errors and clean up the code. This was already a good introduction to JAX and showed me that for the next project I need to spend a lot of time on the JAX code. 

---

### Surprises & "Aha" Moments

The biggest surprise was seeing that my HMC sampler was slower in terms of ESS/sec than my MCMC sampler. HMC's short autocorrelation length was overwhelmed by the computation of calculating the finite difference gradient.

---

### AI Usage Reflection
**Most Significant AI Interaction This Project:**
I used AI in this project for writing my function definitions and for helping with error debugging. 

**Critical Thinking Check:**
AI was a big help for helping me understand the JAX code and navigate the errors. It was a tutor for me to get through the JAX documentation.

---

### What Got Me Excited
I was excited to work with the corner plot! I always see these in papers, and this was my first time looking into how to make one. I was very happy with how easy it was to get my end result.

---

### What I Want to Explore Further
I want to explore JAX further, which I'm excited to have the chance to do so within the next project.

---

### What would you tell yourself if you were starting this project again?
Always remember to print and plot first! 