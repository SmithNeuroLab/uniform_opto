[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10892426.svg)](https://doi.org/10.5281/zenodo.10892426)

# LE/LI recurrent network model

## Description

A recurrent network linear rate model with local excitation/lateral inhibition (LE/LI, also known as Mexican Hat or Difference of Gaussian) 
connectivity, used to model the effect of stimulating the network with different input patterns. 

## Getting Started

### Dependencies
The code found here uses the following python libraries:

* For running simulations: NumPy (1.22.3), SciPy (1.7.3)
* For visualizing results (optional): Matplotlib, Scikit-image

### Installing

* All code and required files can be found in the 'linearLELI' folder. Import paths may need to be amended to 
reflect where 'linearLELI' is saved on your personal device.

### Executing program

* runModel.runModel is the primary function to call to run a simulation. If 'structured_input' is provided, 
the model will generate activity driven by this input. In the absence of an structured pattern, model produces 
activity patterns driven by uniform, noisy inputs.

* This model includes a heterogeneity parameter (h), which modifies the strength of local perturbations 
to LE/LI connectivity, with h=0 indicating an isotropic, non-perturbed connectivity, and h=1 indicating strong perturbation.

* For a demonstration on how to run a model simulation for various input patterns, see the jupyter notebook 'linearLELI_demo.ipynp'



## License

This project is licensed under the MIT License - see the LICENSE.md file for details
 
