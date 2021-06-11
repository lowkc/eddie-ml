import numpy as np
from pyscf.tools.cubegen import Cube
from pyscf import gto, scf, mp, df, dft
from pyscf import lib
from ase.io import read
import os
from glob import glob

def read_cube(cube_file):
    with open(cube_file, 'r') as f:
        f.readline() # comment1
        f.readline() # comment2
        data = f.readline().split() # number of atoms and origin
        natm = int(data[0])
        boxorig = np.array([float(x) for x in data[1:]]) # origin point
        def parse_nx(data):
            d = data.split()
            return int(d[0]), np.array([float(x) for x in d[1:]])
        nx, xs = parse_nx(f.readline())
        ny, ys = parse_nx(f.readline())
        nz, zs = parse_nx(f.readline())
        atoms = []
        for ia in range(natm):
            d = f.readline().split()
            atoms.append([int(d[0]), [float(x) for x in d[2:]]])
        mol = gto.M(atom=atoms, unit='Bohr')
        data = f.read()
        
    cube_data = np.array([float(x) for x in data.split()])
    cube_data_3D = cube_data.reshape(nx, ny, nz)
    dct = {'natm' : natm, 'origin' : boxorig, 'cube_data':cube_data,
           'cube_data_3D': cube_data_3D,
           'nx':nx, 'xs':xs, 'ny':ny, 'ys':ys, 'nz':nz, 'zs':zs, 'mol':mol}
        
    return dct

def xyz_to_Mol(filepath, basis='cc-pVTZ', n=0, charge=0):
    '''
    Reads in an extended XYZ file for an xyz file and returns the nth monomer.
    Specify n=0 or 1, and charge for charged species.
    '''
    xyzfile = read(str(filepath), '{}'.format(n))
    symb = xyzfile.get_chemical_symbols()
    coords = xyzfile.get_positions()
    natoms = len(coords)
    atoms = []
    for i in range(natoms):
        coord = coords[i]
        atoms.append([symb[i], (coord[0], coord[1], coord[2])])
    
    mol = gto.M(atom=atoms, basis=basis, charge=charge)

    return mol

def dimerxyz_to_Mol(filepath, basis='cc-pVTZ', charge=0):
    '''
    Reads in an extended XYZ file for an intermolecular dimer
    and converts it to a PySCF object. 
    '''
    xyzfile = read(str(filepath), ':')
    geom = xyzfile[0] + xyzfile[1]
    symb = geom.get_chemical_symbols()
    coords = geom.get_positions()
    natoms = len(coords)
    atoms = []
    for i in range(natoms):
        coord = coords[i]
        atoms.append([symb[i], (coord[0], coord[1], coord[2])])
    
    mol = gto.M(atom=atoms, basis=basis, charge=charge)

    return mol

def get_charges(xyzfile, SSIloc='path/to/SSI_unCP/files'):
    '''
    Assigns total charge, mono1, and mono2 charges for systems in the SSI database for xyzfiles.
    '''
    basename = os.path.basename(xyzfile)
    if str(basename).startswith('S66'):
        return 0,0,0
    elif str(basename).startswith('C'):
        return 1,-1,0
    else:
        systemname = basename.split(r'-d')[0]
        monomers = sorted(glob(os.path.join(SSIloc, '{}*'.format(systemname))))

        if len(monomers) != 2:
            raise ValueError("Check the number of monomers for {}".format(basename))

        charges = []

        for monomer in monomers:
            with open(monomer) as f:
                lines = f.readlines()
                charges.append(lines[1].split(" ")[0])

        monoAcharge = int(charges[0])
        monoBcharge = int(charges[1])
        tot_charge = int(charges[0]) + int(charges[1])
        return monoAcharge, monoBcharge, tot_charge

def write_single_cube(filename, resolution=0.1, write_cube=True, charge=0):
    molecule = dimerxyz_to_Mol(filename)
    mf = scf.RHF(molecule)
    mf.kernel()
    dm = mf.make_rdm1(ao_repr=True)
    
    # Generate cube dimensions
    orig = generate_uniform_grid(molecule, spacing=resolution, rotate=False, verbose=False)[1]
    cube = Cube(molecule, resolution=resolution, margin=4, origin=orig)
    nx = cube.nx
    ny = cube.ny
    nz = cube.nz
    box = np.diag(cube.box)
    blksize = min(8000, cube.get_ngrids())
    rho = np.empty(cube.get_ngrids())
    for ip0, ip1 in lib.prange(0, cube.get_ngrids(), blksize):
        ao = molecule.eval_gto('GTOval', cube.get_coords()[ip0:ip1])
        rho[ip0:ip1] = dft.numint.eval_rho(molecule, ao, dm)
    rho = rho.reshape([nx, ny, nz])

    if write_cube:
        basename = os.path.basename(filename)[:-4]
        cube.write(rho, '{}.cube'.format(basename),
            comment='Cube_vector {}'.format(box))

    
def dimer_cube_difference(filename, method, resolution=0.1, extension=5.0, charges=None, write_cube=False, path=None):
    '''
    Reads in a dimer XYZ and prints out a .cube file containing
    the electron density difference between the individual monomers and the dimer.
    Choice of HF, MP2, or PBE0 density with a cc-pVTZ basis set.
    The default resolution is set to 0.1 Bohr with a 5.0 Angstrom extension on all sides.
    '''
    if charges is None:
        mon1_charge, mon2_charge, tot_charge = get_charges(filename)
    else: 
        mon1_charge, mon2_charge, tot_charge = charges 
    dimer = dimerxyz_to_Mol(filename, charge=tot_charge)
    mono1 = xyz_to_Mol(filename, n=0, charge=mon1_charge)
    mono2 = xyz_to_Mol(filename, n=1, charge=mon2_charge)
    if path == None:
        path = os.getcwd()

    if method not in ['HF', 'MP2', 'PBE0']:
        raise ValueError('Methods currently implemented: HF, MP2, PBE0 only.')
    # Get density matrices
    if method == 'HF':
        dim_mf = scf.RHF(dimer)
        dim_mf.kernel()
        dimer_dm = dim_mf.make_rdm1(ao_repr=True)

        m1_mf = scf.RHF(mono1)
        m1_mf.kernel()
        mono1_dm = m1_mf.make_rdm1(ao_repr=True)

        m2_mf = scf.RHF(mono2)
        m2_mf.kernel()
        mono2_dm = m2_mf.make_rdm1(ao_repr=True)

    elif method == 'MP2':
        dim_mf = scf.RHF(dimer)
        dim_mf.kernel()
        dim_mp2 = mp.MP2(dim_mf)
        dim_mp2.kernel()
        dimer_dm = dim_mp2.make_rdm1(ao_repr=True)

        m1_mf = scf.RHF(mono1)
        m1_mf.kernel()
        m1_mp2 = mp.MP2(m1_mf)
        m1_mp2.kernel()
        mono1_dm = m1_mp2.make_rdm1(ao_repr=True)

        m2_mf = scf.RHF(mono2)
        m2_mf.kernel()
        m2_mp2 = mp.MP2(m2_mf)
        m2_mp2.kernel()
        mono2_dm = m2_mp2.make_rdm1(ao_repr=True)

    elif method == 'PBE0':
        dim_mf = dimer.KS()
        dim_mf.xc = 'pbe0'
        dim_mf.kernel()
        dimer_dm = dim_mf.make_rdm1(ao_repr=True)

        m1_mf = mono1.KS()
        m1_mf.xc = 'pbe0'
        m1_mf.kernel()
        mono1_dm = m1_mf.make_rdm1(ao_repr=True)

        m2_mf = mono2.KS()
        m2_mf.xc = 'pbe0'
        m2_mf.kernel()
        mono2_dm = m2_mf.make_rdm1(ao_repr=True)

    # Generate the cube dimensions based on the shape of the dimer
    orig = generate_uniform_grid(dimer, spacing=resolution, rotate=False, verbose=False)[1]
    # Set origin to that based on a uniform grid
    dimer_cube = Cube(dimer, resolution=resolution, margin=extension, origin=orig)
    nx = dimer_cube.nx
    ny = dimer_cube.ny
    nz = dimer_cube.nz
    box = np.diag(dimer_cube.box)
    print(box)
    blksize = min(8000, dimer_cube.get_ngrids()) # ngrids is the same for all 3 Mols
    # Dimer density
    dimer_rho = np.empty(dimer_cube.get_ngrids())
    for ip0, ip1 in lib.prange(0, dimer_cube.get_ngrids(), blksize):
        ao = dimer.eval_gto('GTOval', dimer_cube.get_coords()[ip0:ip1])
        dimer_rho[ip0:ip1] = dft.numint.eval_rho(dimer, ao, dimer_dm)
    dimer_rho = dimer_rho.reshape(nx, ny, nz)
    
    # Monomer densities
    mono1_cube = Cube(mono1, nx, ny, nz, margin=extension, origin=orig, extent=box)
    mono1_rho = np.empty(mono1_cube.get_ngrids())
    for ip0, ip1 in lib.prange(0, mono1_cube.get_ngrids(), blksize):
        ao = mono1.eval_gto('GTOval', mono1_cube.get_coords()[ip0:ip1])
        mono1_rho[ip0:ip1] = dft.numint.eval_rho(mono1, ao, mono1_dm)
    mono1_rho = mono1_rho.reshape(nx, ny, nz)

    mono2_cube = Cube(mono2, nx, ny, nz, margin=extension, origin=orig, extent=box)
    mono2_rho = np.empty(mono2_cube.get_ngrids())
    for ip0, ip1 in lib.prange(0, mono2_cube.get_ngrids(), blksize):
        ao = mono2.eval_gto('GTOval', mono2_cube.get_coords()[ip0:ip1])
        mono2_rho[ip0:ip1] = dft.numint.eval_rho(mono2, ao, mono2_dm)
    mono2_rho = mono2_rho.reshape(nx, ny, nz)
    
    rho_diff = dimer_rho - (mono1_rho + mono2_rho)

    if write_cube:
        basename = os.path.basename(filename)[:-4]
        dimer_cube.write(rho_diff, '{}/{}.cube'.format(path, basename),
            comment='{}/cc-pvTZ density difference for {} dimer'.format(method, basename))

    return rho_diff


def generate_uniform_grid(molecule, spacing=0.2, extension=4, rotate=False, verbose=True):
    '''
    molecule = a pySCF mol object
    spacing = the increment between grid points
    extension = the amount to extend the cube on each side of the molecule
    rotate = when True, the molecule is rotated so the axes of the cube file
    are aligned with the principle axes of rotation of the molecule.
    '''
    numbers = molecule.atom_charges()
    pseudo_numbers = molecule.atom_charges()
    coordinates = molecule.atom_coords()
    # calculate the centre of mass of the nuclear charges
    totz = np.sum(pseudo_numbers)
    com = np.dot(pseudo_numbers, coordinates) / totz
    
    if rotate:
        # calculate moment of inertia tensor:
        itensor = np.zeros([3,3])
        for i in range(pseudo_numbers.shape[0]):
            xyz = coordinates[i] - com
            r = np.linalg.norm(xyz)**2.0
            tempitens = np.diag([r,r,r])
            tempitens -= np.outer(xyz.T, xyz)
            itensor += pseudo_numbers[i] * tempitens
        _, v = np.linalg.eigh(itensor)
        new_coords = np.dot((coordinates - com), v)
        axes = spacing * v
        
    else:
        # use the original coordinates
        new_coords = coordinates
        # compute the unit vectors of the cubic grid's coordinate system
        axes = np.diag([spacing, spacing, spacing])
        
    # max and min value of the coordinates
    max_coordinate = np.amax(new_coords, axis=0)
    min_coordinate = np.amin(new_coords, axis=0)
    # compute the required number of points along each axis
    shape = (max_coordinate - min_coordinate + 2.0*extension) / spacing
    shape = np.ceil(shape)
    shape = np.array(shape, int)
    origin = com - np.dot((0.5*shape), axes)
    
    npoints_x, npoints_y, npoints_z = shape
    npoints = npoints_x * npoints_y * npoints_z # total number of grid points
    
    points = np.zeros((npoints, 3)) # array to store coordinates of grid points
    coords = np.array(np.meshgrid(np.arange(npoints_x), np.arange(npoints_y),
                                np.arange(npoints_z)))
    coords = np.swapaxes(coords, 1, 2)
    coords = coords.reshape(3, -1)
    coords = coords.T
    points = coords.dot(axes)
    # compute coordinates of grid points relative to the origin
    points += origin

    if verbose:
        print('Cube origin: {}'.format(origin))
    
    return points, origin
