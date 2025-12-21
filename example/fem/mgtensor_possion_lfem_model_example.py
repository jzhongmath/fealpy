import argparse

from fealpy.backend import bm
from fealpy.mesh import TriangleMesh, IntervalMesh

from fealpy.fem.mgtensor_possion_lfem_model import MGTensorPossionLFEMModel
from fealpy.fem.backup.mgtensor_possion_lfem_model import MGTensorPossionLFEMModelOld

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        任意次拉格朗日有限元方法求解possion方程
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help="默认后端为 numpy. 还可以选择 pytorch, jax, tensorflow 等")

parser.add_argument('--n',
        default=55, type=int,
        help='Degree of Lagrange finite element space, default is 2.')

parser.add_argument('--level',
        default=4, type=int,
        help='Degree of Lagrange finite element space, default is 2.')

options = vars(parser.parse_args())

# 610, 23 s
# 1000, 70 s
# 1800, 120 s
# 3500, 270 s
# 4800, 400 s

from fealpy.backend import bm

bm.set_backend(options['backend'])
n = options['n']
level = options['level']
tmesh = TriangleMesh.from_box([0, 1, 0, 1], nx=n, ny=n)
# imesh = IntervalMesh.from_interval_domain([0, 1], nx=2*(level - 1)*n)
imesh = IntervalMesh.from_interval_domain([0, 1], nx=20)

model = MGTensorPossionLFEMModel(options=options)
# model = MGTensorPossionLFEMModelOld(options=options)

model.set_pde(2)
model.set_mesh(tmesh, imesh)
model.set_space_degree(1)
model.run()
