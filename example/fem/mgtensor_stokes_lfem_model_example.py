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

parser.add_argument('--init_point',
    default = (0.0, 0.0), type = tuple,
    help = "Initial point for chip positioning.")

parser.add_argument('--chip_height',
    default = 1, type = float,
    help = "Height of the microfluidic chip.")

parser.add_argument('--inlet_length',
    default = 0, type = float,
    help = "Length of the inlet section.")

parser.add_argument('--outlet_length',
    default = 0, type = float,
    help = "Length of the outlet section.")

parser.add_argument('--radius',
    default = 1 / (1 * 5), type = float,
    help = "Radius of the pillars.")

parser.add_argument('--n_rows',
    default = 1, type = int,
    help = "Number of rows of pillars in each stage.")

parser.add_argument('--n_cols',
    default = 1, type = int,
    help = "Number of columns of pillars in each stage.")

parser.add_argument('--tan_angle',
    default = 0, type = float,
    help = "Tangent of the deflection angle.")

parser.add_argument('--n_stages',
    default = 1, type = int,
    help = "Number of stages (or periods) in the chip.")

parser.add_argument('--stage_length',
    default = 7, type = float,
    help = "Number of stages (or periods) in the chip.")

parser.add_argument('--lc',
    default = 0.3/2, type = float,
    help = "Grid size for meshing.")

parser.add_argument('--show_figure',
    default = False, type = bool,
    help = "Whether to display the generated mesh.")

parser.add_argument('--space_degree',
        default=2, type=int,
        help='Degree of Lagrange finite element space, default is 2.')

parser.add_argument('--n',
        default=15, type=int,
        help='Degree of Lagrange finite element space, default is 2.')

parser.add_argument('--level',
        default=2, type=int,
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

from fealpy.fem import (
    MGTensorStokesLFEMModel, 
    MGTensorStokesLFEMModelOLD,
    MGTensorStokesLFEMModelI
)
from fealpy.geometry import DLDMicrofluidicChipModeler
from fealpy.mesher import DLDMicrofluidicChipMesher

import gmsh

options = vars(parser.parse_args())

# bm.set_backend('pytorch')
# 0.087  16 124s -> 88s
# 0.072  22 
# 0.046  54
# 0.0419 64
#
bm.set_backend('numpy'); options['lc'] = 1/10

# bm.set_default_device('cuda')

gmsh.initialize()
modeler = DLDMicrofluidicChipModeler(options)
modeler.build(gmsh)
mesher = DLDMicrofluidicChipMesher(options)
mesher.generate(modeler, gmsh)
# gmsh.fltk.run()
gmsh.finalize()
# import ipdb;ipdb.set_trace()
# imesh = IntervalMesh.from_interval_domain([0, 1], nx=2**4)
# model = MGTensorStokesLFEMModelI(options=options)
# model.set_init_mesher(mesher.mesh, imesh, n=0)
# model.set_pde(3)
# model.run()

"""
确保三角形与区间网格加密次数一致：
    n = refine + (level - 1)

n = 1, refine = 1
n = 2, refine = 2
n = 3, refine = 3
n = 4, refine = 3, level += 1
n = 5, refine = 3, level += 2
n = 6, refine = 3, level += 3

"""

k = 3
step = 1 + 7
err = []
refine = [1, 2, 3, 3, 3]
level =  [0, 0, 0, 1, 2]

for n in range(1,step):
    # imesh = IntervalMesh.from_interval_domain([0, 0.1], nx=2*(level - 1)*n)
    imesh = IntervalMesh.from_interval_domain([0, 1], nx=2*2**n)

    model = MGTensorStokesLFEMModelI(options=options)
    from fealpy.mesh import TriangleMesh
    node = bm.array([[0.0, 0.0], 
                      [1.0, 0.0], 
                      [1.0, 1.0], 
                      [0.0, 1.0],
                      [0.3, 0.2]], dtype=bm.float64)
    cell = bm.array([[0, 1, 4],
                     [1, 2, 4],
                     [2, 3, 4],
                     [3, 0, 4]], dtype= bm.int32)
    mesh = TriangleMesh(node, cell)

    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=1, ny=1)
    # import ipdb;ipdb.set_trace()
    model.set_init_mesher(mesh, imesh, n=refine[n-1], level=level[n-1])
    model.set_pde(k)
    err.append(model.run())

    if n > 1:
        err1 = bm.array(err)
        print(err1)
        print(bm.log(err1[:-1]/err1[1:])/bm.log(2))

