import argparse

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        张量网格上 Stokes 方程的求解
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help="默认后端为 numpy. 还可以选择 pytorch, jax, tensorflow 等")

parser.add_argument('--thickness',
    default = 0.1, type = float,
    help = "thickness.")


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

parser.add_argument('--lc',
    default = 0.2/2, type = float,
    help = "Grid size for meshing.")

parser.add_argument('--space_degree',
        default=2, type=int,
        help='Degree of Lagrange finite element space, default is 2.')

parser.add_argument('--level',
        default=4, type=int,
        help='Degree of Lagrange finite element space, default is 2.')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())


from fealpy.backend import bm
from fealpy.mesh import IntervalMesh, TensorPrismMesh

from fealpy.fem import WPRLFEMModel
from fealpy.mesher import WPRMesher

import gmsh

options = vars(parser.parse_args())

# bm.set_backend('pytorch')
bm.set_backend('numpy'); options['lc'] = 2
# bm.set_default_device('cuda')

mesher = WPRMesher(options)
mesher.generate()

level = options['level']
imesh = IntervalMesh.from_interval_domain([0, 0.1], nx=3)

model = WPRLFEMModel(options=options)
model.set_init_mesher(mesher, imesh)
model.set_inlet_condition()
model.run()
