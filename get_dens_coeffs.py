'''
Code to process all .cube files in a folder and output density coefficients.
The target folder should contain all .cube files, plus a corresponding data file containing
the interaction energy target prediction value named as follows:
water.cube
water.dat
The .cube and .dat extensions can be specified.
Option to use ipyparallel for parallel processing.
'''

import sys, os, glob
import time
from density.utils import get_view, preprocess_all
import argparse

parser = argparse.ArgumentParser(description='Get the deformation density coefficients for all deformation density .cube files in a folder.')
parser.add_argument('--outputname', type=str, default='output', help='Name of output file.')
parser.add_argument('--path', type=str, default=os.getcwd(), help='Path containing cube files.')
parser.add_argument('--cube_ext', type=str, default='cube', help='Cube file extension.')
parser.add_argument('--dat_ext', type=str, default='dat', help='Data file extension.')
start = time.time()

# Modify this basis set for coefficients as needed - currently, all elements have the same number
# of basis functions
default_basis = {'r_o_o': 2.5, 'r_i_o': 0.0,
              'r_o_h': 2.5, 'r_i_h': 0.0,
              'r_o_c': 2.5, 'r_i_c': 0.0,
              'r_o_n': 2.5, 'r_i_n': 0.0,
              'n_rad_c': 4, 'n_rad_o': 4,
              'n_rad_h': 4, 'n_rad_n': 4,
              'n_l_o': 3, 'n_l_h': 3,
              'n_l_c': 3, 'n_l_n': 3,
              'gamma_o': 0, 'gamma_h': 0,
              'gamma_c': 0, 'gamma_n': 0}


args = parser.parse_args()

if __name__ == "__main__":
    preprocess_all(args.path, name=args.outputname, basis=default_basis, dens_ext=args.cube_ext, eng_ext=args.dat_ext)
