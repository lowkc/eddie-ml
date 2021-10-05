import sys, re
import os, glob
import numpy as np
import pandas as pd

"""
Run this to get the list of atom numbers and atom types required to run EDDIE.
Usage: python get_atoms_and_atomtypes.py /path/to/xyzdir/ file.energies
where file.energies is the output of running get_dens_coeffs and /path/to/xyzdir/ contains the .xyz files for the processed systems.
"""

xyzfolder = sys.argv[1]
energies = pd.read_csv(sys.argv[2], header=None)

energies['Natoms'] = 0
energies['AtomTypes'] = np.empty((len(energies), 0)).tolist()
pattern = r'\s*-?[0-9]*\.[0-9]*'
pattern = pattern + pattern + pattern

for i, fname in enumerate(energies[1]):
    file = os.path.join(xyzfolder, fname[:-4]) + '.xyz'
    with open(file) as f:
        atom_list = []
        first_atom_count = f.readline()
        for line in f:
            if re.search('[A-Z][a-z]?' + pattern, line):
                line = line.strip()
                line = line.split()
                atom_list.append(line[0])
        energies.set_value(i, 'AtomTypes', atom_list)
        energies.set_value(i, 'Natoms', first_atom_count)

fname = str(sys.argv[2]).split('.')[0]
natoms = energies['Natoms'].to_numpy()
np.save('{}_atoms.npy'.format(fname), natoms)
atom_types = energies['AtomTypes'].to_numpy()
np.save('{}_atomtypes.npy'.format(fname), atom_types)

