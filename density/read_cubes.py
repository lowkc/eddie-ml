# Utility functions for real-space grid properties extracted from .cube files
import numpy as np
import struct
from ase.units import Bohr
from density import Density
from ase import Atoms
import re
import os

def _read_cube_header(f):
    # Read the title
    title = f.readline().strip()
    # skip the second line
    f.readline()

    def read_grid_line(line):
        """Read a grid line from the cube file"""
        words = line.split()
        return (
            int(words[0]),
            np.array([float(words[1]), float(words[2]), float(words[3])], float)
            # all coordinates in a cube file are in atomic units
        )

    # number of atoms and origin of the grid
    natom, origin = read_grid_line(f.readline())
    # numer of grid points in A direction and step vector A, and so on
    shape0, axis0 = read_grid_line(f.readline())
    shape1, axis1 = read_grid_line(f.readline())
    shape2, axis2 = read_grid_line(f.readline())
    shape = np.array([shape0, shape1, shape2], int)
    axes = np.array([axis0, axis1, axis2])

    cell = np.array(axes*shape.reshape(-1,1))
    grid = shape

    def read_coordinate_line(line):
        """Read an atom number and coordinate from the cube file"""
        words = line.split()
        return (
            int(words[0]), float(words[1]),
            np.array([float(words[2]), float(words[3]), float(words[4])], float)
            # all coordinates in a cube file are in atomic units
        )

    numbers = np.zeros(natom, int)
    pseudo_numbers = np.zeros(natom, float)
    coordinates = np.zeros((natom, 3), float)
    for i in range(natom):
        numbers[i], pseudo_numbers[i], coordinates[i] = read_coordinate_line(f.readline())
        # If the pseudo_number field is zero, we assume that no effective core
        # potentials were used.
        if pseudo_numbers[i] == 0.0:
            pseudo_numbers[i] = numbers[i]

    return origin, coordinates, numbers, cell, grid


def _read_cube_data(f, grid):
    data = np.zeros(tuple(grid), float)
    tmp = data.ravel()
    counter = 0
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        words = line.split()
        for word in words:
            tmp[counter] = float(word)
            counter += 1
    return data


def load_cube(filename, threshold=1e-15):
    '''Load data from a cube file
       **Arguments:**
       filename
            The name of the cube file
       **Returns** a dictionary with ``title``, ``coordinates``, ``numbers``,
       ``cube_data``, ``grid``, ``pseudo_numbers``.
       threshold makes it so that any value smaller than the cutoff is treated as 0.
    '''
    with open(filename) as f:
        origin, coordinates, numbers, cell, grid = _read_cube_header(f)
        data = _read_cube_data(f, grid)
        if threshold:
            data = (np.abs(data) > threshold) * data
        return Density(data, cell*Bohr, grid, origin*Bohr)


def get_atoms(cubefile):
    '''
    Find atomic data in .cube file
    Note: the positions of the atoms are translated by the origin of the cube.
    '''
    with open(cubefile) as f:
        origin, coordinates, numbers, cell, grid = _read_cube_header(f)
    coordinates *= Bohr # Get atom positions in Ang
    origin *= Bohr
    atoms = Atoms(numbers, coordinates)
    atoms.translate(np.abs(origin))
    return atoms


def get_atoms_and_dens(filename, threshold=1e-30):
    with open(filename) as f:
        origin, coordinates, numbers, cell, grid = _read_cube_header(f)
        data = _read_cube_data(f, grid)
    if threshold:
        data = (np.abs(data) > threshold) * data

    atoms = Atoms(numbers, coordinates*Bohr)
    atoms.translate(np.abs(origin*Bohr))
        
    return atoms, Density(data, cell*Bohr, grid, origin*Bohr)


def get_density(filepath):
    '''
    Import data from a .cube file (or similar real-space grid file).
    Data is saved in global variables.

    Parameters:
        filepath: str
            path to .cube file containing density

    Returns:
        Density (class)
    '''
    unitcell = np.zeros([3,3]) # assuming this is the origin in the cube file

    with open(filepath, 'r') as rhofile:
        # unit cell is in Bohrs in file - convert to Angstrom

        rhofile.readline() # skip 1st line
        rhofile.readline() # skip 2nd line
        #unitcell = rhofile.readline().split()[1:] 
        #unitcell = np.array([float(unitcell[0][1:]), float(unitcell[1]), float(unitcell[2][:-1])])
        #unitcell = np.diag(unitcell)

        data = rhofile.readline().split()
        natoms = int(data[0])
        boxorig = np.array([float(x) for x in data[1:]]) # origin point

        def parse_nx(data):
            d = data.split()
            return int(d[0]), np.array([float(x) for x in d[1:]])
        nx, xs = parse_nx(rhofile.readline())
        ny, ys = parse_nx(rhofile.readline())
        nz, zs = parse_nx(rhofile.readline())

        unit_cell_dims = np.diag(np.array([np.multiply(nx, xs[0]), np.multiply(ny, ys[1]), np.multiply(nz, zs[2])]))
        # these are the REAL unit cell dimensions - but cube is not centred around this point
        
        grid = np.array([nx, ny, nz])
        unit_cell_adj = np.zeros((3,3))

        for i in range(3):
            unit_cell_adj[i,:] = np.absolute(unit_cell_dims[i,:]/boxorig[i])
        #print(unit_cell_adj)
        
        for i in range(natoms):
            d = rhofile.readline()  # skip the lines containing atomic data
        data = rhofile.read()

        rho = np.array([float(x) for x in data.split()])
        rho_reshaped = rho.reshape([nx, ny, nz])

    return Density(rho_reshaped, unit_cell_adj*Bohr, grid, boxorig*Bohr)  # returns unit cell in Angstrom



def get_energy(outputfile, keywords=['IE_SRSMP2']):
    '''
    Find output energy values specified by keywords in results file.
    '''
    assert isinstance(keywords, (list, tuple))
    values = []
    with open(outputfile, 'r') as file:
        for keyword in keywords:
            file.seek(0)
            p = re.compile(keyword + r'=.*-?\d*.?\d*')
            p_wo = re.compile(keyword + r'=\s*')
            content = file.read()
            withnumber = p.findall(content)[0]
            wonumber = p_wo.findall(content)[0]
            values.append(float(withnumber[len(wonumber):]))
            values.append(os.path.basename(outputfile))
    
    return values
