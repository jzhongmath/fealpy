
import argparse

from fealpy.backend import bm
from fealpy.mesher import WPRMesher


parser = argparse.ArgumentParser(description=
    """
    Test microfluidic chip geometry modeling.
    This script generates the geometry for a microfluidic chip using the specified parameters.
    """)

parser.add_argument('--backend',
    default = 'numpy', type = str,
    help = "Default backend is numpy. You can also choose pytorch, jax, tensorflow, etc.")

parser.add_argument('--block',
        default= {
                'length': 6.0,
                'width': 2.0
        }, 
        help='Default backend is numpy')

parser.add_argument('--inlet',
        default= {
                'length': 0.5,
                'width': 0.5
        }, 
        help='Default backend is numpy')

parser.add_argument('--gap',
    default = 0.1, type = float,
    help = "Radius of the pillars.")

parser.add_argument('--h',
    default = 0.06, type = float,
    help = "Grid size for meshing.")

parser.add_argument('--return_mesh',
    default = True, type = bool,
    help = "Whether to display the generated mesh.")

parser.add_argument('--show_figure',
    default = True, type = bool,
    help = "Whether to display the generated mesh.")


options = vars(parser.parse_args())
bm.set_backend(options['backend'])
mesher = WPRMesher(options)
mesher.generate()

