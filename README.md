# Machine learning prediction of long-range intermolecular interaction energy based on electron deformation density
This repository contains the code needed to reproduce results in the paper, "Machine learning prediction of long-range intermolecular interaction energy based on electron deformation density" (K. Low, M. L. Coote, and E. I. Izgorodina, 2021).

The code in this repository references the MLCF code developed by Dick, Sebastian, and Marivi Fernandez-Serra in "Learning from the density to correct total energy and forces in first principle simulations." The Journal of Chemical Physics 151.14 (2019): 144102. See: https://github.com/semodi/mlcf to download and for a full list of requirements.

## Requirements
Requires Python version 3.6 or later. Necessary packages are ``ase``, ``pyscf``, ``scipy``, and ``sympy``. 

## Usage
All structures are in the `data` folder. Codes can be found in the `model` folder.

### Generate a deformation density
```
python deformation_density.py xyzfile.xyz
```

### Get coefficients from density cube files
```
python get_dens_coeffs.py 
```
