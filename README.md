# Electron deformation density interaction energy machine learning (EDDIE-ML) code
This repository contains the code needed to reproduce results in the paper, "Machine learning prediction of dimer interaction energies benefits from combining quantum mechnical information with atom-based descriptors" (K. Low, M. L. Coote, and E. I. Izgorodina, 2021).

The code in this repository references the MLCF code developed by Dick, Sebastian, and Marivi Fernandez-Serra in "Learning from the density to correct total energy and forces in first principle simulations." The Journal of Chemical Physics 151.14 (2019): 144102. See: https://github.com/semodi/mlcf to download and for a full list of requirements.

## Requirements
Requires Python version 3.6 or later. Necessary packages are ``ase``, ``pyscf``, ``dscribe``, ``scipy``, and ``sympy``. 

## Usage
All structures are in the `data` folder. Codes can be found in the `model` folder. Kernels for use in GPR models are in the `kernels` folder.

### Generate a deformation density
```
python deformation_density.py xyzfile.xyz
```

### Get coefficients from density cube files
Run the following code in a folder containing one or more .cube files. Basis set parameters such as atom types and radial cutoffs can be specified within the script.
```
python get_dens_coeffs.py 
```
