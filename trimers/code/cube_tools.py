# LOW RESOLUTION CUBE TOOLS
import numpy as np
from pyscf.tools.cubegen import Cube
from pyscf import gto, scf, mp, df, dft
from pyscf import dftd3
from pyscf import lib
from ase.io import read
import os
from glob import glob
#from mpi4pyscf import scf as mpi_scf

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

def xyz_to_Mol(filepath, basis='cc-pVTZ', n=0, charge=0, spin=0):
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
    
    mol = gto.M(atom=atoms, basis=basis, charge=charge, spin=spin)

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

def trimerxyz_to_Mol(filepath, basis='cc-pVTZ', charge=0):
    '''
    Reads in an extended XYZ file for an intermolecular trimer
    and converts it to a PySCF object. 
    '''
    xyzfile = read(str(filepath), ':')
    geom = xyzfile[0] + xyzfile[1] + xyzfile[2]
    symb = geom.get_chemical_symbols()
    coords = geom.get_positions()
    natoms = len(coords)
    atoms = []
    for i in range(natoms):
        coord = coords[i]
        atoms.append([symb[i], (coord[0], coord[1], coord[2])])
    
    mol = gto.M(atom=atoms, basis=basis, charge=charge)

    return mol

def dimerxyz_to_ghost_mols(filepath, basis='cc-pVTZ', charges=[0,0]):
    '''Read in an extended XYZ file for an intermolecular dimer and return two PySCF mole objects.'''
    xyzfile = read(str(filepath), ':')
    geom1, geom2 = xyzfile[0], xyzfile[1]
    symb1 = geom1.get_chemical_symbols()
    symb2 = geom2.get_chemical_symbols()
    coords1 = geom1.get_positions()
    coords2 = geom2.get_positions()
    natoms1 = len(coords1)
    natoms2 = len(coords2)
    mol1normal = []
    mol2normal = []

    for i in range(natoms1):# Loop over the atoms in molecule 1
        # Molecule 1 is a ghost, molecule 2 is normal
        coord1 = coords1[i]
        mol1normal.append(['ghost-'+symb1[i], (coord1[0], coord1[1], coord1[2])])
        mol2normal.append([symb1[i], (coord1[0], coord1[1], coord1[2])])
    for j in range(natoms2):
        # Molecule 2 is a ghost, molecule 1 is normal
        coord2 = coords2[j]
        mol1normal.append([symb2[j], (coord2[0], coord2[1], coord2[2])])
        mol2normal.append(['ghost-'+symb2[j], (coord2[0], coord2[1], coord2[2])])

    mol1 = gto.M(atom=mol1normal, basis=basis, charge=charges[0], verbose=0)
    mol2 = gto.M(atom=mol2normal, basis=basis, charge=charges[0], verbose=0)
    return mol1, mol2


def get_charges(xyzfile):
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
        orig_loc = '/home/klow12/sn29/klow/ML/DBs/xyzfiles/SSI_xyzfiles/unCPfiles/' # monomers path
        monomers = sorted(glob(os.path.join(orig_loc, '{}*'.format(systemname))))

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

def write_trimer_cube(filename, path=None, resolution=0.2, write_cube=True, charge=0):
    if path == None:
        path = os.getcwd()

    molecule = xyz_to_Mol(filename)

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
        cube.write(rho, '{}/{}.cube'.format(path, basename),
            comment='Cube_vector {}'.format(box))

def write_single_cube(filename, resolution=0.2, write_cube=True, charge=0):
        #xyzfile = read(str(filename), '0')
    #xyzfile.center(about=0.)
    #symb = xyzfile.get_chemical_symbols()
    #coords = xyzfile.get_positions()
    #natoms = len(coords)
    #atoms = []
    #for i in range(natoms):
    #    coord = coords[i]
    #    atoms.append([symb[i], (coord[0], coord[1], coord[2])])
    molecule = dimerxyz_to_Mol(filename)
    #molecule = gto.M(atom=atoms, basis='cc-pvtz', charge=charge)

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


def hf_edefdens_RI_parallel(filename, resolution=0.2, extension=4.0, write_cube=False, path=None, charges=None):
    '''
    Read in dimer XYZ file and prints out a .cube file containing
    the electron density difference between the individual monomers and the dimer.
    The default resolution is 0.2 Bohr.
    The charge of the dimer and monomer should be specified if they are not 0.
    Uses mpirun for parallelisation.
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

    # MPI parallel HF calculation
    dim_df = df.DF(dimer).build()
    dim_mf = mpi_scf.RHF(dimer).density_fit()
    dim_mf.with_df = dim_df
    dim_mf.kernel()
    dimer_dm = dim_mf.make_rdm1(ao_repr=True)

    m1_df = df.DF(mono1).build()
    m1_mf = mpi_scf.RHF(mono1).density_fit()
    m1_mf.with_df = m1_df
    m1_mf.kernel()
    mono1_dm = m1_mf.make_rdm1(ao_repr=True)

    m2_df = df.DF(mono2).build()
    m2_mf = mpi_scf.RHF(mono2).density_fit()
    m2_mf.with_df = m2_df
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
    
def df_deformation_density(filename, method, resolution=0.2, extension=4.0, write_cube=False, path=None, charges=None):
    '''
    Read in dimer XYZ file and prints out a .cube file containing
    the electron density difference between the individual monomers and the dimer.
    The default resolution is 0.2 Bohr.
    The charge of the dimer and monomer should be specified if they are not 0.
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
        dim_mf = scf.RHF(dimer).density_fit()
        dim_mf.only_dfj=True
        dim_mf.kernel()
        dimer_dm = dim_mf.make_rdm1(ao_repr=True)

        m1_mf = scf.RHF(mono1).density_fit()
        m1_mf.only_dfj=True
        m1_mf.kernel()
        mono1_dm = m1_mf.make_rdm1(ao_repr=True)

        m2_mf = scf.RHF(mono2).density_fit()
        m2_mf.only_dfj=True
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

def FMO_deformdens(filename, method, resolution=0.2, extension=4.0, write_cube=False, path=None, charges=[0,0,0,0]):
    '''
    Read in trimer XYZ file and prints out a .cube file containing
    the electron density difference between the individual monomers and the trimer.
    The default resolution is 0.2 Bohr.
    The charge of the trimer and monomer should be specified if they are not 0.
    '''
    m1_charge, m2_charge, m3_charge, tot_charge = charges
    trimer = trimerxyz_to_Mol(filename, charge=tot_charge)
    mono1 = xyz_to_Mol(filename, n=0, charge=m1_charge)
    mono2 = xyz_to_Mol(filename, n=1, charge=m2_charge)
    mono3 = xyz_to_Mol(filename, n=2, charge=m3_charge)
    dimer = mono2+mono3

    if path == None:
        path = os.getcwd()

    if method not in ['HF', 'MP2', 'PBE0', 'REVPBE']:
        raise ValueError('Methods currently implemented: HF, MP2, PBE0, REVPBE-D3.')
    
    # TODO: add in other methods later, currently only interested in HF.
    # Get density matrices
    if method == 'HF':
        tri_mf = scf.RHF(trimer).density_fit()
        tri_mf.kernel()
        tri_dm = tri_mf.make_rdm1(ao_repr=True)

        m1_mf = scf.RHF(mono1).density_fit()
        m1_mf.kernel()
        mono1_dm = m1_mf.make_rdm1(ao_repr=True)

        m2_mf = scf.RHF(mono2).density_fit()
        m2_mf.kernel()
        mono2_dm = m2_mf.make_rdm1(ao_repr=True)

        m3_mf = scf.RHF(mono3).density_fit()
        m3_mf.kernel()
        mono3_dm = m3_mf.make_rdm1(ao_repr=True)

        dim_mf = scf.RHF(dimer).density_fit()
        dim_mf.kernel()
        dim_dm = dim_mf.make_rdm1(ao_repr=True)

    
    # Generate the cube dimensions based on the shape of the dimer
    orig = generate_uniform_grid(trimer, spacing=resolution, rotate=False, verbose=False)[1]
    # Set origin to that based on a uniform grid
    trimer_cube = Cube(trimer, resolution=resolution, margin=extension, origin=orig)
    nx = trimer_cube.nx
    ny = trimer_cube.ny
    nz = trimer_cube.nz
    box = np.diag(trimer_cube.box)
    blksize = min(8000, trimer_cube.get_ngrids()) # ngrids is the same for all 3 Mols

    # Trimer density
    trimer_rho = np.empty(trimer_cube.get_ngrids())
    for ip0, ip1 in lib.prange(0, trimer_cube.get_ngrids(), blksize):
        ao = trimer.eval_gto('GTOval', trimer_cube.get_coords()[ip0:ip1])
        trimer_rho[ip0:ip1] = dft.numint.eval_rho(trimer, ao, tri_dm)
    trimer_rho = trimer_rho.reshape(nx, ny, nz)
    
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

    mono3_cube = Cube(mono3, nx, ny, nz, margin=extension, origin=orig, extent=box)
    mono3_rho = np.empty(mono3_cube.get_ngrids())
    for ip0, ip1 in lib.prange(0, mono3_cube.get_ngrids(), blksize):
        ao = mono3.eval_gto('GTOval', mono3_cube.get_coords()[ip0:ip1])
        mono3_rho[ip0:ip1] = dft.numint.eval_rho(mono3, ao, mono3_dm)
    mono3_rho = mono3_rho.reshape(nx, ny, nz)

    # Dimer density
    dimer_cube = Cube(dimer, nx, ny, nz, margin=extension, origin=orig, extent=box)
    dimer_rho = np.empty(dimer_cube.get_ngrids())
    for ip0, ip1 in lib.prange(0, dimer_cube.get_ngrids(), blksize):
        ao = dimer.eval_gto('GTOval', dimer_cube.get_coords()[ip0:ip1])
        dimer_rho[ip0:ip1] = dft.numint.eval_rho(dimer, ao, dim_dm)
    dimer_rho = dimer_rho.reshape(nx, ny, nz)
    
    rho_diff = trimer_rho - mono1_rho - (dimer_rho - mono2_rho - mono3_rho)

    if write_cube:
        basename = os.path.basename(filename)[:-4]
        trimer_cube.write(rho_diff, '{}/{}.cube'.format(path, basename),
            comment='{}/cc-pvTZ density difference for {} dimer'.format(method, basename))

    return rho_diff


def trimer_deformdens(filename, method, resolution=0.2, extension=4.0, write_cube=False, path=None, charges=[0,0,0,0]):
    '''
    Read in trimer XYZ file and prints out a .cube file containing
    the electron density difference between the individual monomers and the trimer.
    The default resolution is 0.2 Bohr.
    The charge of the trimer and monomer should be specified if they are not 0.
    '''
    m1_charge, m2_charge, m3_charge, tot_charge = charges
    trimer = trimerxyz_to_Mol(filename, charge=tot_charge)
    mono1 = xyz_to_Mol(filename, n=0, charge=m1_charge)
    mono2 = xyz_to_Mol(filename, n=1, charge=m2_charge)
    mono3 = xyz_to_Mol(filename, n=2, charge=m3_charge)

    if path == None:
        path = os.getcwd()

    if method not in ['HF', 'MP2', 'PBE0', 'REVPBE']:
        raise ValueError('Methods currently implemented: HF, MP2, PBE0, REVPBE-D3.')
    
    # TODO: add in other methods later, currently only interested in HF.
    # Get density matrices
    if method == 'HF':
        tri_mf = scf.RHF(trimer).density_fit()
        tri_mf.kernel()
        tri_dm = tri_mf.make_rdm1(ao_repr=True)

        m1_mf = scf.RHF(mono1).density_fit()
        m1_mf.kernel()
        mono1_dm = m1_mf.make_rdm1(ao_repr=True)

        m2_mf = scf.RHF(mono2).density_fit()
        m2_mf.kernel()
        mono2_dm = m2_mf.make_rdm1(ao_repr=True)

        m3_mf = scf.RHF(mono3).density_fit()
        m3_mf.kernel()
        mono3_dm = m3_mf.make_rdm1(ao_repr=True)
    
    elif method == "PBE0":
        tri_mf = trimer.KS()
        tri_mf.xc = 'pbe0'
        tri_mf.kernel()
        tri_dm = tri_mf.make_rdm1(ao_repr=True)

        m1_mf = mono1.KS()
        m1_mf.xc = 'pbe0'
        m1_mf.kernel()
        mono1_dm = m1_mf.make_rdm1(ao_repr=True)

        m2_mf = mono2.KS()
        m2_mf.xc = 'pbe0'
        m2_mf.kernel()
        mono2_dm = m2_mf.make_rdm1(ao_repr=True)

        m3_mf = mono3.KS()
        m3_mf.xc = 'pbe0'
        m3_mf.kernel()
        mono3_dm = m3_mf.make_rdm1(ao_repr=True)

    # Generate the cube dimensions based on the shape of the dimer
    orig = generate_uniform_grid(trimer, spacing=resolution, rotate=False, verbose=False)[1]
    # Set origin to that based on a uniform grid
    trimer_cube = Cube(trimer, resolution=resolution, margin=extension, origin=orig)
    nx = trimer_cube.nx
    ny = trimer_cube.ny
    nz = trimer_cube.nz
    box = np.diag(trimer_cube.box)
    blksize = min(8000, trimer_cube.get_ngrids()) # ngrids is the same for all 3 Mols

    # Trimer density
    trimer_rho = np.empty(trimer_cube.get_ngrids())
    for ip0, ip1 in lib.prange(0, trimer_cube.get_ngrids(), blksize):
        ao = trimer.eval_gto('GTOval', trimer_cube.get_coords()[ip0:ip1])
        trimer_rho[ip0:ip1] = dft.numint.eval_rho(trimer, ao, tri_dm)
    trimer_rho = trimer_rho.reshape(nx, ny, nz)
    
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

    mono3_cube = Cube(mono3, nx, ny, nz, margin=extension, origin=orig, extent=box)
    mono3_rho = np.empty(mono3_cube.get_ngrids())
    for ip0, ip1 in lib.prange(0, mono3_cube.get_ngrids(), blksize):
        ao = mono3.eval_gto('GTOval', mono3_cube.get_coords()[ip0:ip1])
        mono3_rho[ip0:ip1] = dft.numint.eval_rho(mono3, ao, mono3_dm)
    mono3_rho = mono3_rho.reshape(nx, ny, nz)
    
    rho_diff = trimer_rho - mono1_rho - mono2_rho - mono3_rho

    if write_cube:
        basename = os.path.basename(filename)[:-4]
        trimer_cube.write(rho_diff, '{}/{}.cube'.format(path, basename),
            comment='{}/cc-pvTZ density difference for {} dimer'.format(method, basename))

    return rho_diff


def ghost_deformdens(filename, method, resolution=0.2, extension=4.0, write_cube=False, path=None, charges=None):
    '''
    Read in dimer XYZ file and prints out a .cube file containing
    the electron density difference between the individual monomers and the dimer.
    The default resolution is 0.2 Bohr.
    The charge of the dimer and monomer should be specified if they are not 0.

    HF ONLY FOR NOW!
    '''
    if charges is None:
        mon1_charge, mon2_charge, tot_charge = get_charges(filename)
    else:
        mon1_charge, mon2_charge, tot_charge = charges
    dimer = dimerxyz_to_Mol(filename, charge=tot_charge)
    mono1, mono2 = dimerxyz_to_ghost_mols(filename, basis='cc-pVTZ', charges=[mon1_charge, mon2_charge])

    if path == None:
        path = os.getcwd()

    dim_mf = scf.RHF(dimer).density_fit()
    dim_mf.kernel()
    dimer_dm = dim_mf.make_rdm1(ao_repr=True)

    m1_mf = scf.RHF(mono1).density_fit()
    m1_mf.kernel()
    mono1_dm = m1_mf.make_rdm1(ao_repr=True)

    m2_mf = scf.RHF(mono2).density_fit()
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


def deformdens_RI(filename, method, resolution=0.2, extension=4.0, write_cube=False, path=None, charges=None):
    '''
    Read in dimer XYZ file and prints out a .cube file containing
    the electron density difference between the individual monomers and the dimer.
    The default resolution is 0.2 Bohr.
    The charge of the dimer and monomer should be specified if they are not 0.
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

    if method not in ['HF', 'MP2', 'PBE0', 'REVPBE']:
        raise ValueError('Methods currently implemented: HF, MP2, PBE0, REVPBE-D3.')
    # Get density matrices
    if method == 'HF':
        dim_mf = scf.RHF(dimer).density_fit()
        dim_mf.kernel()
        dimer_dm = dim_mf.make_rdm1(ao_repr=True)

        m1_mf = scf.RHF(mono1).density_fit()
        m1_mf.kernel()
        mono1_dm = m1_mf.make_rdm1(ao_repr=True)

        m2_mf = scf.RHF(mono2).density_fit()
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

    elif method =='REVPBE':
        dim_mf = dft.RKS(dimer)
        dim_mf.xc = 'REVPBE'
        dim_mf_d3 = dftd3.dftd3(dim_mf)
        dim_mf_d3.kernel()
        dimer_dm = dim_mf_d3.make_rdm1(ao_repr=True)

        m1_mf = dft.RKS(mono1)
        m1_mf.xc = 'REVPBE'
        m1_mf_d3 = dftd3.dftd3(m1_mf)
        m1_mf_d3.kernel()
        mono1_dm = m1_mf_d3.make_rdm1(ao_repr=True)

        m2_mf = dft.RKS(mono2)
        m2_mf.xc = 'REVPBE'
        m2_mf_d3 = dftd3.dftd3(m2_mf)
        m2_mf_d3.kernel()
        mono2_dm = m2_mf_d3.make_rdm1(ao_repr=True)

    # Generate the cube dimensions based on the shape of the dimer
    orig = generate_uniform_grid(dimer, spacing=resolution, rotate=False, verbose=False)[1]
    # Set origin to that based on a uniform grid
    dimer_cube = Cube(dimer, resolution=resolution, margin=extension, origin=orig)
    nx = dimer_cube.nx
    ny = dimer_cube.ny
    nz = dimer_cube.nz
    box = np.diag(dimer_cube.box)
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


    
def dimer_cube_difference(filename, method, resolution=0.2, extension=4.0, write_cube=False, path=None, charges=None):
    '''
    Read in dimer XYZ file and prints out a .cube file containing
    the electron density difference between the individual monomers and the dimer.
    Basis set is set to cc-pVTZ.
    The default resolution is 0.2 Bohr.
    The charge of the dimer and monomer should be specified if they are not 0.
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
        if dim_mf.converged == False:
            print('WARNING! Dimer SCF not converged for {}'.format(filename))
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

def pbe_cubes(filename, method='PBE0', resolution=0.2, extension=4.0, write_cube=False, path=None, charges=None):
    '''
    Read in dimer XYZ file and prints out a .cube file containing
    the electron density difference between the individual monomers and the dimer.
    The default resolution is 0.2 Bohr.
    The charge of the dimer and monomer should be specified if they are not 0.
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
    dft.numint.NumInt.libxs = dft.xcfun

    if method == 'PBE0':
        dimer.verbose = 4
        dim_mf = dimer.KS()
        dim_mf._numint.libxc = dft.xcfun
        dim_mf.xc = 'pbe0'
#        if dim_mf.converged == False:
#            dim_mf.level_shift = 0.5
#            dim_mf.kernel()#        if dim_mf.converged == False:
        if dim_mf.converged == False:
            dim_mf = dim_mf.newton()
            dim_mf.kernel()
#        if dim_mf.converged == False:
#            dim_mf.damp = 0.5
#            dim_mf.level_shift = 0.3
#            dim_mf.diis_start_cycle = 2
#            dim_mf.newton()
#            dim_mf.kernel()
        if dim_mf.converged == False:
            raise Exception('Error! Dimer SCF not converged for {}'.format(filename))
        dimer_dm = dim_mf.make_rdm1(ao_repr=True)

        mono1.verbose = 4
        m1_mf = mono1.KS()
        m1_mf._numint.libxc = dft.xcfun
        m1_mf.xc = 'pbe0'
        m1_mf = m1_mf.newton()
        m1_mf.kernel()
        mono1_dm = m1_mf.make_rdm1(ao_repr=True)

        mono2.verbose = 4
        m2_mf = mono2.KS()
        m2_mf._numint.libxc = dft.xcfun
        m2_mf.xc = 'pbe0'
        m2_mf = m2_mf.newton()
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

def get_density_and_deriv(molecule, tot_charge=0):
    '''
    This script reads in a dimer XYZ returns the HF/cc-pvTZ density and the electron density gradient.
    THIS IS WRONG: FIX IT
    '''
    mol = dimerxyz_to_Mol(filename, charge=tot_charge)
    grids = generate_uniform_grid(mol, rotate=False)
    mf = scf.RHF(mol)
    mf.kernel()
    dm = mf.make_rdm1(ao_repr=True)
    ao_value = pyscf.dft.numint.eval_ao(mol, grids, deriv=1)
    rho = pyscf.dft.numint.eval_rho(mol, ao_value, dm, xctype='LDA')
    
    dct = {'density':rho[0], 'gradient':rho[1:4], 'hessian':rho[4:10].reshape(3,3)}
    return dct
    

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
