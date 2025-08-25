# Vascular Draining Confounds Laminar Decoding in fMRI  

by **Jonas Karolis Degutis, Denis Chaimow, Romy Lorenz**  

---

## Overview  

This repository contains the full simulation pipeline for exploring vascular draining confounds in laminar decoding with fMRI.  
It provides scripts to reproduce the analyses and figures from the study.  

---

## How to Run  

All simulations and analyses are implemented in **Python**. The main simulation functions are housed in `simulation.py`, while specific scripts reproduce the figures.  

### Scripts and Their Outputs  

- **`runSimulationParallel.py`**  
  - Simulates and plots data for:  
    - **Figure 1c** (using `layers=3`)  
    - **Figure 2a** (using `layers=3`)  
    - **Figure 5b** (using `layers=6`)  

- **`runSimulation_twoPatterns.py`**  
  - Simulates and plots data for **Figure 2b** (using `layers=3`) 

- **`runDeconvolution.py`**  
  - Deconvolves simulated data and plots the results for **Figure 1b** and **Figure 2b** (using `layers=3`)
  - And **Figure 5d** (using `layers=3`)

- **`runSimulation_samePattern.py`**  
  - Simulates and plots data for **Figure 3**  

- **`runVoxMisalignment.py`** and **`runVascModelMisalignment.py`**  
  - Run analyses of misalignment effects, plotted in **Figure 4**  

- **`simulation.py`**  
  - Core simulation class containing the main functions used across all scripts.  

---

## Requirements  

- Python 3.x  
- Standard scientific Python stack (NumPy, SciPy, Matplotlib)  

(All dependencies found in `requirements.txt`)  

---

## Contact  

For questions, please contact:  
📧 **j.karolis.degutis(at)gmail.com**  

---

## License  

MIT License  

© J. Karolis Degutis, 2025  
