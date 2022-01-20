# Electron deformation density interaction energy machine learning (EDDIE-ML) code
This repository contains the code needed to reproduce results in the paper, "Inclusion of more physics leads to less data: Learning the interaction energy as a function of electron deformation density with limited training data" (K. Low, M. L. Coote, and E. I. Izgorodina, 2021).

The code in this repository references the MLCF code developed by Dick, Sebastian, and Marivi Fernandez-Serra in "Learning from the density to correct total energy and forces in first principle simulations." The Journal of Chemical Physics 151.14 (2019): 144102. See: https://github.com/semodi/mlcf to download and for a full list of requirements.

## Requirements
Requires Python version 3.6 or later. Necessary packages are ``ase``, ``pyscf``, ``dscribe``, ``scipy``, ``spherical_functions`` and ``sympy``. 

## Usage
All structures are in the `data` folder. Codes can be found in the `model` folder. Kernels for use in GPR models are in the `kernels` folder.

### Generate a deformation density
```
python get_deformation_density.py xyzfile.xyz
```

### Get coefficients from density cube files
Run the following code in a folder containing one or more .cube files. Basis set parameters such as atom types and radial cutoffs can be specified within the script.
```
python get_dens_coeffs.py 
```
### Run EDDIE trained on small neutral dimers
To reproduce results from the neutral dimer dataset or predict for your own molecules, load the saved model:
```
import pickle

with open('data/EDDIE_neutraldimer_model.pkl', 'rb') as f:
    model = pickle.load(file)

model.predict(X, y, atoms, atomtypes)
```
where ```atoms``` contains the number of atoms in the first dimer of X, and atom types is a list containing the elements of X. These can be accessed from ```get_atoms_and_atomtypes.py```.
