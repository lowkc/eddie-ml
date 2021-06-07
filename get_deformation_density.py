from enum import auto
import numpy as np
import argparse
from density import cube_utils

parser = argparse.ArgumentParser(description='Get the deformation density for any dimer .xyz structure.')
parser.add_argument('custom_dimer', action='store_true')
parser.add_argument('charges', nargs=3, help='If using a custom dimer (i.e., not from SSI, S66, or IL), add list for [monomer1 charge, monomer2 charge, total charge].')
parser.add_argument('cube_file', type=str, help='Name of cube file.')
parser.add_argument('method', type=str, help='Density type. Choose from HF, MP2, or PBE0')
parser.add_argument('--resolution', type=float, default=0.1, help='.cube resolution')
parser.add_argument('--extension', type=float, default=5, help='Extension on sides of .cube file')

args = parser.parse_args()

if __name__ == "__main__":
    if args.custom_dimer:
        mon1_charge, mon2_charge, tot_charge = args.charges
        cube_utils.dimer_cube_difference(args.cube_file, args.method, resolution=args.resolution,
        extension=args.extension, charges=[mon1_charge, mon2_charge, tot_charge], auto_charges=False,
        write_cube=True)
    else:
        cube_utils.dimer_cube_difference(args.cube_file, args.method, resolution=args.resolution,
        extension=args.extension, auto_charges=True, write_cube=True)
