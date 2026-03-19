      
from typing import Optional, Union
from fealpy.backend import bm

from fealpy.typing import TensorLike
from fealpy.decorator import variantmethod, cartesian

from fealpy.mesh import TriangleMesh, IntervalMesh
from fealpy.functionspace import LagrangeFESpace, functionspace
from fealpy.fem import BilinearForm, LinearForm, DirichletBC, BlockForm, LinearBlockForm
from fealpy.fem import (
        ScalarDiffusionIntegrator, ScalarMassIntegrator, 
        PressWorkIntegrator, CouplingMassIntegrator,
        SourceIntegrator, ScalarSourceIntegrator, 
    )

from fealpy.model import PDEModelManager, ComputationalModel

from fealpy.mesher import DLDMicrofluidicChipMesher

from fealpy.sparse import spdiags, coo_matrix, csr_matrix, COOTensor, CSRTensor
from fealpy.solver import cg, gmres, spsolve, transferP1red, transferP2red, StokesLSCDGS, indofP1, indofP2
from fealpy.utils import timer

import scipy.sparse as sp
import scipy.sparse.linalg as lg

from petsc4py import PETSc

import time
import gc

"""
ModelI, 用于验证纯Box区域收敛阶, 已成功

u1 =   sin(πx)cos(πy)cos(πz)
u2 =   cos(πx)sin(πy)cos(πz)
u3 = -2cos(πx)cos(πy)sin(πz)

we have ∇⋅u = 0. Take 
    p(x, y, z) = sin(2πx) + cos(2πy) + sin(2πz),
and
    ∇p = (2πcos(2πx), -2πsin(2πy), 2πcos(2πz)),

we have
    f = -Δu + ∇p = 3π^2u + ∇p.
"""

def csr_to_petsc_mat(csr):
    nrows, ncols = csr.shape
    mat = PETSc.Mat().createAIJ(size=(nrows, ncols),
                        csr=(csr.indptr, csr.indices, csr.data))
    mat.assemble()
    return mat


class SumOperator:
    def __init__(self, *ops):
        self.ops = ops
        self.shape = ops[0].shape

    def __matmul__(self, x):
        y = 0
        for op in self.ops:
            y = y + (op @ x)
        return y


class LinearOperator:
    def __matmul__(self, x):
        raise NotImplementedError
    
    def __add__(self, other):
        return SumOperator(self, other)

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)


class KronOperator(LinearOperator):
    def __init__(self, A, B, num=1):
        self.A = A
        self.B = B
        self.num = num
        self.n0, self.m0 = A.shape
        self.n1, self.m1 = B.shape
        self.n = self.m0 * self.m1
        self.shape = (num*self.n0*self.n1, num*self.m0*self.m1)

    def __matmul__(self, x):
        v = bm.copy(x)
        A = self.A
        B = self.B

        if self.num == 3:
            U1 = bm.reshape(v[:self.n], (self.m0, self.m1))
            U2 = bm.reshape(v[self.n:2*self.n], (self.m0, self.m1))
            U3 = bm.reshape(v[2*self.n:3*self.n], (self.m0, self.m1))

            Y1 = A @ U1 @ B
            Y2 = A @ U2 @ B
            Y3 = A @ U3 @ B
            Y = bm.concat([Y1.ravel(), Y2.ravel(), Y3.ravel()], axis=0)
            return Y
        elif self.num == 1:
            X = bm.reshape(v, (self.m0, self.m1))
            Y = A @ X @ B
            Y = Y.ravel()
        return Y


class StokesOperator(LinearOperator):
    def __init__(self, Ax, Mx, Az, Mz, Bx, Bz, Mx_, Mz_):
        self.Ax = Ax
        self.Mx = Mx
        self.Az = Az
        self.Mz = Mz
        self.Bx = Bx
        self.Bz = Bz
        self.Mx_ = Mx_
        self.Mz_ = Mz_
        self.set_up()

    def set_up(self):
        self.n_Ax = self.Ax.shape[0]
        self.n_Mz = self.Mz.shape[0]

        self.n_Bx = self.Bx.shape[0]
        self.m_Bx = self.Bx.shape[1]

        self.n_Mz_ = self.Mz_.shape[0]
        self.m_Mz_ = self.Mz_.shape[1]

        self.n_Mx_ = self.Mx_.shape[0]
        self.m_Mx_ = self.Mx_.shape[1]

        self.n_Bz = self.Bz.shape[0]
        self.m_Bz = self.Bz.shape[1]

        self.n_u0 = self.n_Ax * self.n_Mz
        self.n_p = self.n_Bx * self.n_Mz_
        self.n_A = 3 * self.n_u0 + self.n_p
        self.shape = self.n_A, self.n_A

    def assembly(self):
        pass

    def __matmul__(self, x):
        v = bm.copy(x)

        U1 = bm.reshape(v[:self.n_u0], (self.n_Ax, self.n_Mz))
        U2 = bm.reshape(v[self.n_u0:2*self.n_u0], (self.n_Ax, self.n_Mz))
        U3 = bm.reshape(v[2*self.n_u0:3*self.n_u0], (self.n_Ax, self.n_Mz))

        U4 = bm.reshape(v[:2*self.n_u0], (self.m_Bx, self.m_Mz_))
        U5 = bm.reshape(v[2*self.n_u0:3*self.n_u0], (self.m_Mx_, self.m_Bz))

        P = bm.reshape(v[-self.n_p:], (self.n_Bx, self.n_Mz_))
        
        U1 = bm.to_numpy(U1)
        U2 = bm.to_numpy(U2)
        U3 = bm.to_numpy(U3)
        U4 = bm.to_numpy(U4)
        U5 = bm.to_numpy(U5)
        P = bm.to_numpy(P)

        AU1 = (self.Mz @ (self.Ax @ U1).T).T  + (self.Az @ (self.Mx @ U1).T).T
        AU2 = (self.Mz @ (self.Ax @ U2).T).T  + (self.Az @ (self.Mx @ U2).T).T
        AU3 = (self.Mz @ (self.Ax @ U3).T).T  + (self.Az @ (self.Mx @ U3).T).T

        BP1 = (self.Mz_.T @ (self.Bx.T @ P).T).T
        BP2 = (self.Bz.T @ (self.Mx_.T @ P).T).T
        
        BU1 = (self.Mz_ @ (self.Bx @ U4).T).T
        BU2 = (self.Bz @ (self.Mx_ @ U5).T).T
        
        l1 = bm.concat([bm.tensor(AU1.ravel()), bm.tensor(AU2.ravel())], axis=0) + bm.tensor(BP1.ravel())
        l2 = bm.tensor(AU3.ravel()) + bm.tensor(BP2.ravel())
        l3 = bm.tensor(BU1.ravel()) + bm.tensor(BU2.ravel())

        y = bm.concat([l1, l2, l3], axis=0)

        return y


class A0iOperator(LinearOperator):
    def __init__(self, Ax, Mx, Mz, Az):
        self.Ax = Ax
        self.Mx = Mx
        self.Mz = Mz
        self.Az = Az
        self.bigMz = sp.block_diag([Mz, Mz, Mz])
        self.bigAz = sp.block_diag([Az, Az, Az])

        self.m0, self.n0 = Ax.shape
        self.m1, self.n1 = Mz.shape
        self.shape = (self.m0*self.m1, self.n0*self.n1)

    def set_up(self):
        pass
    
    @variantmethod('direct')
    def assembly(self):
        A0 = sp.kron(sp.tril(A=self.Ax, k=-1), self.Mz, format='csr') + \
             sp.kron(sp.tril(A=self.Mx, k=-1), self.Az, format='csr') + \
             sp.kron(sp.diags(self.Ax.diagonal()), sp.tril(A=self.Mz), format='csr') + \
             sp.kron(sp.diags(self.Mx.diagonal()), sp.tril(A=self.Az), format='csr')
        
        # A0 = sp.block_diag([A0,A0,A0], format='csr')
        # A0_dense = sp.kron(self.Ax, self.Mz) + sp.kron(self.Mx, self.Az)
        return A0

    # update
    def __matmul__(self, x):
        n = len(x) // 3
        v = bm.copy(x)
        X = bm.reshape(v, (self.n0, self.m1))
        Y = self.Ax @ X @ self.Mz + self.Mx @ X @ self.Az
        Y = Y.ravel()
        return Y
        # U1 = bm.reshape(v[:n], (self.n0, self.m1))
        # U2 = bm.reshape(v[n:2*n], (self.n0, self.m1))
        # U3 = bm.reshape(v[2*n:3*n], (self.n0, self.m1))
        # U = bm.concat([U1,U2,U3], axis=1)
        n = len(x) // 3
        v = bm.copy(x)
        import ipdb;ipdb.set_trace()
        # 直接计算每个块，避免构建大矩阵 U 和 full matrix multiplication
        Y1 = self.Ax @ (bm.reshape(v[:n], (self.n0, self.m1)) @ self.Mz) + self.Mx @ (bm.reshape(v[:n], (self.n0, self.m1)) @ self.Az)
        Y2 = self.Ax @ (bm.reshape(v[n:2*n], (self.n0, self.m1)) @ self.Mz) + self.Mx @ (bm.reshape(v[n:2*n], (self.n0, self.m1)) @ self.Az)
        Y3 = self.Ax @ (bm.reshape(v[2*n:3*n], (self.n0, self.m1)) @ self.Mz) + self.Mx @ (bm.reshape(v[2*n:3*n], (self.n0, self.m1)) @ self.Az)
        return [Y1.ravel(), Y2.ravel(), Y3.ravel()]
        # y = ([Y[:,:self.m1].ravel(),
        #                Y[:,self.m1:2*self.m1].ravel(),
        #                Y[:,2*self.m1:3*self.m1].ravel()])
        return y

class AiOperator(LinearOperator):
    def __init__(self, Ax, Mx, Mz, Az):
        self.Ax = Ax
        self.Mx = Mx
        self.Mz = Mz
        self.Az = Az

        self.m0, self.n0 = Ax.shape
        self.m1, self.n1 = Mz.shape
        self.n_u0 = self.m0*self.m1
        self.shape = (3*self.m0*self.m1, 3*self.n0*self.n1)

    def set_up(self):
        pass

    def assembly(self):
        A0_dense = sp.kron(self.Ax, self.Mz) + sp.kron(self.Mx, self.Az)
        A_dense = sp.block_diag((A0_dense, A0_dense, A0_dense))
        return A_dense
    
    def __matmul__(self, x):
        v = bm.copy(x)
        U1 = bm.reshape(v[:self.n_u0], (self.n0, self.m1))
        U2 = bm.reshape(v[self.n_u0:2*self.n_u0], (self.n0, self.m1))
        U3 = bm.reshape(v[2*self.n_u0:3*self.n_u0], (self.n0, self.m1))

        Y1 = self.Ax @ U1 @ self.Mz + self.Mx @ U1 @ self.Az
        Y2 = self.Ax @ U2 @ self.Mz + self.Mx @ U2 @ self.Az
        Y3 = self.Ax @ U3 @ self.Mz + self.Mx @ U3 @ self.Az
        Y = bm.concat([Y1.ravel(), Y2.ravel(), Y3.ravel()], axis=0)
        return Y


class BiOperator(LinearOperator):
    def __init__(self, Bx, Mx_, Mz_, Bz):
        self.Bx = Bx
        self.Mx_ = Mx_
        self.Mz_ = Mz_
        self.Bz = Bz
        self.Mz_t = Mz_.T
        self.Bzt = Bz.T

        self.n_Bx = self.Bx.shape[0]
        self.m_Bx = self.Bx.shape[1]

        self.n_Mz_ = self.Mz_.shape[0]
        self.m_Mz_ = self.Mz_.shape[1]

        self.n_Mx_ = self.Mx_.shape[0]
        self.m_Mx_ = self.Mx_.shape[1]

        self.n_Bz = self.Bz.shape[0]
        self.m_Bz = self.Bz.shape[1]

        self.m0, self.n0 = Mx_.shape
        self.m1, self.n1 = Bz.shape
        self.n_u0 = self.n0*self.n1
        self.shape = (self.m0*self.m1, 3*self.n0*self.n1)

    def set_up(self):
        pass

    def assembly(self):
        B0 = sp.kron(self.Bx, self.Mz_)
        B1 = sp.kron(self.Mx_, self.Bz)
        B = sp.bmat([[B0, B1]])
        return B

    def __matmul__(self, x):
        v = bm.copy(x)
        U1 = bm.reshape(v[:2*self.n_u0], (self.m_Bx, self.m_Mz_))
        U2 = bm.reshape(v[2*self.n_u0:], (self.m_Mx_, self.m_Bz))

        BU1 = self.Bx @ U1 @ self.Mz_t
        BU2 = self.Mx_ @ U2 @ self.Bzt
        y = BU1.ravel() + BU2.ravel()
        return y


class BtiOperator(LinearOperator):
    def __init__(self, Bxt, Mx_t, Mz_, Bz):
        self.Bxt = Bxt
        self.Mx_t = Mx_t
        self.Mz_ = Mz_
        self.Bz = Bz
        self.Mz_t = Mz_.T
        self.Bzt = Bz.T

        self.n0, self.m0 = Mx_t.shape
        self.m1, self.n1 = Bz.shape
        self.shape = (3*self.n0*self.n1, self.m0*self.m1)

    def set_up(self):
        pass

    def assembly(self):
        B0 = sp.kron(self.Bxt, self.Mz_t)
        B1 = sp.kron(self.Mx_t, self.Bzt)
        B = sp.bmat([[B0], [B1]])
        return B

    def __matmul__(self, x):
        v = bm.copy(x)
        P = bm.reshape(v, (self.m0, self.m1))
        BP1 = self.Bxt @ P @ self.Mz_
        BP2 = self.Mx_t @ P @ self.Bz

        y = bm.concat([BP1.ravel(), BP2.ravel()], axis=0)
        return y


class MGTensorStokesLFEMModelI(ComputationalModel):
    """"Multigrid solver for Poisson equations defined on 
            tensor-product grids using the Linear Finite Element Method (LFEM).
    """
    def __init__(self, options: dict = None):
        self.options = options
        super().__init__(
            pbar_log=options['pbar_log'],
            log_level=options['log_level']
        )

        if options is None:
            options = {} 
        
        self.eps = 1e-10
        self.thickness = options.get('thickness', 0.1)
        self.level = options.get('level')

        self.options = options
        self.x0 = options.get('x0', None)
        self.tol = options.get('tol', 1e-8)  
        self.maxIt = options.get('solvermaxit', 200)  
        self.N0 = options.get('N0', 500)
        self.mu = options.get('smoothingstep', 1)
        self.solver = options.get('solver', 'direct')

        self.cycle_type = options.get('cycle_type', 'VCYCLE')
        self.smoothing_times = options.get('smoothing_times', 1)
        self.preconditioner = options.get('preconditioner', 'none')
        self.coarsegridsolver = options.get('coarsegridsolver', 'direct')
        
        self.coarse_time = 0
        self.smoothing_time = 0
        self.coarse_count = 0
        self.smoothing_count = 0

        self.SGS_time = 0
        self.MUL_time = 0

        self.assembly_time = 0
        self.cycle_MUL_time = 0

        self.setup_time = 0
        self.initial_assembly_time = 0

    def set_init_mesher(self, mesh: TriangleMesh, imesh: IntervalMesh, n: int=0, level: int=0):
        """
        Set the initial mesh for the simulation.
        
        Parameters:
            mesh: The computational mesh object
        """
        tmesh = mesh
        # import ipdb;ipdb.set_trace()
        if n > 0:
            tmesh.uniform_refine(n)
            # imesh.uniform_refine(n)
        
        self.mesh0 = TriangleMesh(tmesh.entity('node'), tmesh.entity('cell'))
        self.mesh1 = TriangleMesh(tmesh.entity('node'), tmesh.entity('cell'))
        self.level += level
        tmesh.uniform_refine(self.level-1)
        self.tmesh = tmesh
        self.imesh = imesh
        
        tnode = tmesh.entity('node') # (NN_t, 2)
        inode = imesh.entity('node') # (NN_i, 1)
        tcell = tmesh.entity('cell')

        iNN = imesh.number_of_nodes()
        tNC = tmesh.number_of_cells()
        
        # xy * z
        node = bm.concat([bm.repeat(tnode, inode.shape[0], axis=0), 
                          bm.tile(inode.T, tnode.shape[0]).T], axis=1)
        
        # 按xy方向一层一层排
        all_cell = iNN * tcell[None, :, :] + bm.arange(iNN)[:, None, None]
        all_cell = all_cell.reshape(-1, tcell.shape[1])
        cell = bm.concat([all_cell[:-tNC], all_cell[tNC:]], axis=1)

        s0 = tmesh.entity_measure('cell')
        s1 = imesh.entity_measure('cell')
        self.cm = bm.einsum('i,j->ij', s1, s0).ravel()

        self.node = node
        self.cell = cell

        # import matplotlib.pyplot as plt
        # from fealpy.mesh import TensorPrismMesh
        # mesh = TensorPrismMesh(self.tmesh, self.imesh)
        # mesh = tmesh
        # ipoints = tmesh.interpolation_points(p=1)
        # fig = plt.figure()
        # axes = fig.add_subplot(111)
        # mesh.add_plot(axes)
        # mesh.find_node(axes, node=ipoints, 
        #             showindex=True, color='r', fontsize='10')
        # # tmesh.find_cell(axes, showindex=True, fontsize='35')
        # plt.show()

    def set_pde(self, k: int=3):
        manager = PDEModelManager('stokes')
        self.pde = manager.get_example(k)

    def set_space_degree(self, p: int=2):
        """
        Set the polynomial degree for function spaces
        """
        self.p = p

    @variantmethod
    def linear_system(self):
        """
        Assemble the linear system for the Stokes equations.
        """
        from fealpy.mesh import TensorPrismMesh
        self.mesh = TensorPrismMesh(self.tmesh, self.imesh)
        
        self.uspace = functionspace(self.mesh, ('Lagrange', 2), shape=(3, -1))
        self.pspace = functionspace(self.mesh, ('Lagrange', 1))
        self.uh = self.uspace.function()
        self.int_space0 = LagrangeFESpace(self.imesh, p=1)
        self.int_space1 = LagrangeFESpace(self.imesh, p=2)
        self.tri_space0 = LagrangeFESpace(self.tmesh, p=1)
        self.tri_space1 = LagrangeFESpace(self.tmesh, p=2)

        form00 = BilinearForm(self.tri_space1)
        form00.add_integrator(ScalarDiffusionIntegrator())
        Ax = form00.assembly().to_scipy()

        form01 = BilinearForm(self.tri_space1)
        form01.add_integrator(ScalarMassIntegrator())
        Mx = form01.assembly().to_scipy()

        form02 = BilinearForm(self.int_space1)
        form02.add_integrator(ScalarDiffusionIntegrator())
        Az = form02.assembly().to_scipy()

        form03 = BilinearForm(self.int_space1)
        form03.add_integrator(ScalarMassIntegrator())
        Mz = form03.assembly().to_scipy()

        self.uspace2d = functionspace(self.tmesh, ('Lagrange', 2), shape=(2, -1))
        self.pspace2d = functionspace(self.tmesh, ('Lagrange', 1))

        form10 = BilinearForm((self.pspace2d, self.uspace2d))
        form10.add_integrator(PressWorkIntegrator(coef=-1.0))
        Bx = form10.assembly().to_scipy().T

        form11 = BilinearForm((self.int_space0, self.int_space1))
        form11.add_integrator(CouplingMassIntegrator())
        Mz_ = form11.assembly().to_scipy().T

        self.uspace1d = functionspace(self.imesh, ('Lagrange', 2), shape=(1, -1))
        self.pspace1d = functionspace(self.imesh, ('Lagrange', 1))

        form12 = BilinearForm((self.pspace1d, self.uspace1d))
        form12.add_integrator(PressWorkIntegrator(coef=-1.0))
        Bz = form12.assembly().to_scipy().T

        form13 = BilinearForm((self.tri_space0, self.tri_space1))
        form13.add_integrator(CouplingMassIntegrator())
        Mx_ = form13.assembly().to_scipy().T
        
        self.ugdof = Ax.shape[0]*Mz.shape[0]
        self.total_dof = Ax.shape[0]*Mz.shape[0]*3+Bx.shape[1]*Mz_.shape[1]
        print(f'自由度个数: {Ax.shape[0]*Mz.shape[0]*3+Bx.shape[0]*Mz_.shape[0]}')
        print(f'压力自由度个数：{Bx.shape[0]*Mz_.shape[0]}')
        # import ipdb;ipdb.set_trace()
        op = StokesOperator(Ax, Mx, Az, Mz, Bx, Bz, Mx_, Mz_)
        # import ipdb;ipdb.set_trace()
        A1 = sp.kron(Ax, Mz) + \
             sp.kron(Mx, Az)
        B0 = sp.kron(Bx, Mz_)
        B1 = sp.kron(Mx_, Bz)
        
        A0 = sp.block_diag((A1, A1, A1))
        B = sp.bmat([[B0, B1]])
        A = sp.bmat([[A0, B.T],
                     [B, None]])

        from fealpy.sparse import COOTensor
        A = COOTensor(
            indices=bm.stack([A.row, A.col], axis=0),
            values=A.data,
            spshape=A.shape
        )
        # A = None
        self.n_A = op.n_A
        self.n_p = op.n_p
        self.x0 = bm.zeros((self.n_A,), dtype=bm.float64)
        F = bm.zeros((self.n_A,), dtype=bm.float64)
        F0 = self.assembly_F['notsep']()
        F[:len(F0)] = F0

        return op, A, F
    
    @variantmethod('notsep')
    def assembly_F(self):
        """
        Assembly F on tensor mesh.
        """
        from ..quadrature import (
                GaussLegendreQuadrature, 
                TensorProductQuadrature, 
                TriangleQuadrature
            )
        
        p = 2
        q = p + 3
        qf0 = TriangleQuadrature(q)
        qf1 = GaussLegendreQuadrature(q)

        qf = TensorProductQuadrature((qf0, qf1))
        # bcs: ((NQ0, 3), (NQ1, 2)), ws: (NQ0 * NQ1,)
        bcs, ws = qf.get_quadrature_points_and_weights()

        # compute basis
        raw_phi = [bm.simplex_shape_function(bc, p) for bc in bcs] # ((NQ0, ldof0), (NQ1, ldof1))
        phi = bm.tensorprod(*raw_phi)
        
        # compute source
        ipoints = self.interpolation_points()
        c2f = self.cell_to_ipoint(p=2)
        # idx = bm.arange(18).reshape(-1,6).T.ravel()

        points = bm.einsum('cld,ql->cqd', ipoints[c2f], phi)
        coef_val0 = self.pde.source0(points)
        coef_val1 = self.pde.source1(points)
        coef_val2 = self.pde.source2(points)
        
        # assembly, F0: (gdof,)
        group_tensor = bm.einsum('c, q, cql, cq -> cl', self.cm, ws, phi[None,:], coef_val0)
        F0 = bm.zeros((self.ugdof,),dtype=bm.float64)
        bm.add_at(F0, c2f, group_tensor) # not set_at

        # assembly, F1: (gdof,)
        group_tensor = bm.einsum('c, q, cql, cq -> cl', self.cm, ws, phi[None,:], coef_val1)
        F1 = bm.zeros((self.ugdof,),dtype=bm.float64)
        bm.add_at(F1, c2f, group_tensor) # not set_at

        # assembly, F2: (gdof,)
        group_tensor = bm.einsum('c, q, cql, cq -> cl', self.cm, ws, phi[None,:], coef_val2)
        F2 = bm.zeros((self.ugdof,),dtype=bm.float64)
        bm.add_at(F2, c2f, group_tensor) # not set_at
        F = bm.concat([F0, F1, F2], axis=0)

        return F

    @assembly_F.register('sep')
    def assembly_F(self):
        """
        Assembly F on tensor mesh.
        """
        # assembly F_x
        form0 = LinearForm(self.space0)
        SI0 = ScalarSourceIntegrator(self.sourcex)
        form0.add_integrator(SI0)
        Fx = form0.assembly()

        # assembly F_z
        form1 = LinearForm(self.space1)
        SI1 = ScalarSourceIntegrator(self.sourcez)
        form1.add_integrator(SI1)
        Fz = form1.assembly()
        F = (Fx[:,None]*Fz[None,:]).reshape(-1)
        
        return F 

    def cell_to_ipoint(self, p: int):
        cell = self.cell
        if p == 1:
            return cell[:, [0, 3, 1, 4, 2, 5]]
        tc2i = self.tmesh.cell_to_ipoint(p)
        ic2i = self.imesh.cell_to_ipoint(p)
        iNC = self.imesh.number_of_cells()
        tNC = self.tmesh.number_of_cells()
        igdof = self.imesh.number_of_global_ipoints(p)
        c2i = bm.zeros((iNC * tNC, tc2i.shape[1] * ic2i.shape[1]), dtype=bm.int32)
        idx = bm.arange(tc2i.shape[1] * ic2i.shape[1]).reshape(ic2i.shape[1], tc2i.shape[1]).T.flatten()
        for i in range(iNC):
            c2i[i*tNC:(i+1)*tNC, :] = (igdof* tc2i[None, :, :] + ic2i[i][:, None, None]).transpose(1, 0, 2).reshape(tNC, -1)[:, idx]
        return  c2i
    
    def boundary_dof_index(self):
        isDDof1 = self.tri_space1.is_boundary_dof()
        isDDof2 = self.imesh.boundary_face_flag()
        igdof = self.int_space1.number_of_global_dofs()
        isDDof3 = bm.zeros((igdof, ), dtype=bm.bool)
        bm.set_at(isDDof3, bm.arange(len(isDDof2)), isDDof2)

        bd_dof0 = ~((~isDDof1[:, None]) * (~isDDof3[None, :])).ravel()

        return bd_dof0

    def interpolation_points(self):
        ipoint1 = self.imesh.interpolation_points(p=2)
        ipoint3 = self.tmesh.interpolation_points(p=2)

        p1 = bm.concat([bm.repeat(ipoint3, ipoint1.shape[0], axis=0), 
                          bm.tile(ipoint1.T, (ipoint3.shape[0],)).T], axis=1)
        
        return p1

    def apply_bc(self, op: StokesOperator, F):
        uh = self.x0
        gd = self.pde.velocity_dirichlet
        threshold = self.pde.is_velocity_boundary
        
        dofs = self.boundary_dof_index()
        points = self.interpolation_points() # (2000w, 3) ~ 500 MB
        BdDof = []
        
        index_dof = bm.arange(len(points))[dofs]
        # ipoints: (NI， 3), 边界插值点坐标, 
        bd_point = points[dofs] 
        # flag: (NI,), 判断边界点是否属于某类边界
        flag = threshold(bd_point)
        index_dof = index_dof[flag]
        val = gd(bd_point[flag])

        index_dof = bm.concat([index_dof, index_dof + len(points), 
                            index_dof + 2*len(points)], axis=0)
        val = val.T.reshape(-1)

        BdDof = index_dof
        isBdDof = bm.zeros(self.n_A, dtype=bm.bool)
        isBdDof = bm.set_at(isBdDof, index_dof, True)
        uh = bm.set_at(uh, (..., isBdDof), val)

        F = F - op @ uh # 5000w ~ 400MB
        F = bm.set_at(F, index_dof, uh[index_dof])
        
        # Fixdof
        flag = self.imesh.boundary_face_flag()
        igdof = self.int_space1.number_of_global_dofs()
        isDDof = bm.zeros((igdof, ), dtype=bm.bool)
        bm.set_at(isDDof, bm.arange(len(flag)), flag)
        
        inflag_uz = ~isDDof
        inflag_u = indofP2(self.tmesh, threshold=self.pde.is_velocity_boundary, tensor_mesh=True)

        inflag_u = bm.to_numpy(inflag_u)
        Biginflag_u = bm.to_numpy(bm.concat([inflag_u, inflag_u], axis=0))
        inflag_uz = bm.to_numpy(inflag_uz)

        op.Ax = op.Ax[inflag_u][:,inflag_u]
        op.Mx = op.Mx[inflag_u][:,inflag_u]
        op.Az = op.Az[inflag_uz][:,inflag_uz]
        op.Mz = op.Mz[inflag_uz][:,inflag_uz]

        op.Bx = op.Bx[:,Biginflag_u]
        op.Mx_ = op.Mx_[:,inflag_u]
        op.Mz_ = op.Mz_[:,inflag_uz]
        op.Bz = op.Bz[:,inflag_uz]
        
        op.set_up()
        return op, F, BdDof
       
    def setup(self, op: StokesOperator):
        """Compute restriction and interpolation operators.
        """
        Ax = op.Ax
        Mx = op.Mx
        Az = op.Az
        Mz = op.Mz
        Bx = op.Bx
        Bz = op.Bz
        Mx_ = op.Mx_
        Mz_ = op.Mz_

        level = self.level
        Axi = [None] * level
        Mxi = [None] * level
        Bxi = [None] * level
        Mx_i = [None] * level
        # bigAi = [None] * level
        
        Axi[-1] = Ax
        Mxi[-1] = Mx
        Bxi[-1] = Bx
        Mx_i[-1] = Mx_
        
        # bigAi[-1] = sp.bmat([[A, B.T],[B,None]]).tocsr()
        Nu = bm.zeros((level,), dtype=bm.int32)
        Np = bm.zeros((level,), dtype=bm.int32)
        
        # Compute Pro and Res of u and p.
        Pro_p = transferP1red(self.mesh0, self.level, tensor_mesh=True)
        Pro_u = transferP2red(self.mesh1, self.level, self.pde.is_velocity_boundary, tensor_mesh=True)
        
        for j in range(level - 1, 0, -1):
            Axi[j-1] = Pro_u[j-1].T @ Axi[j] @ Pro_u[j-1]
            Mxi[j-1] = Pro_u[j-1].T @ Mxi[j] @ Pro_u[j-1]

            Bxi[j-1] = Pro_p[j-1].T @ Bxi[j] @ sp.block_diag([Pro_u[j-1],Pro_u[j-1]])
            Mx_i[j-1] = Pro_p[j-1].T @ Mx_i[j] @ Pro_u[j-1]

        self.P_u = [None] * (level-1)
        self.P_p = [None] * (level-1)
        self.R_u = [None] * (level-1)
        self.R_p = [None] * (level-1)

        self.auxMat = [None] * level
        self.A0i = [None] * level
        self.Ai = [None] * level
        self.Bi = [None] * level
        self.Bti = [None] * level
        self.bigAi = [None] * level

        Iz2 = spdiags(bm.ones((op.n_Mz,)), 0, op.n_Mz, op.n_Mz).to_scipy()
        Iz1 = spdiags(bm.ones((op.n_Mz_,)), 0, op.n_Mz_, op.n_Mz_).to_scipy()

        for j in range(self.level):
            self.A0i[j] = A0iOperator(Axi[j], Mxi[j], Mz, Az)
            self.Ai[j] = AiOperator(Axi[j], Mxi[j], Mz, Az)
            self.Bi[j] = BiOperator(Bxi[j], Mx_i[j], Mz_, Bz)
            self.Bti[j] = BtiOperator(Bxi[j].T, Mx_i[j].T, Mz_, Bz)
            Nu[j] = self.A0i[j].shape[0]
            Np[j] = self.Bi[j].shape[0]
            # bigAi[j] = (sp.bmat([[Ai[j], Bi[j].T],[Bi[j], None]]).tocsr())
            if j < self.level - 1:
                self.P_u[j] = KronOperator(Pro_u[j], Iz2, num=3)
                self.P_p[j] = KronOperator(Pro_p[j], Iz1)
                self.R_u[j] = KronOperator(Pro_u[j].T, Iz2, num=3)
                self.R_p[j] = KronOperator(Pro_p[j].T, Iz1)
            
            if j > 0:
                BBt = sp.kron(Bxi[j]@Bxi[j].T, Mz_@Mz_.T) + sp.kron(Mx_i[j]@Mx_i[j].T, Bz@Bz.T)
                # Su = sp.tril(A0)
                Sp = sp.tril(BBt).tocsr()
                Spt = sp.triu(BBt).tocsr()
                DSp = sp.diags_array(1/BBt.diagonal())

                self.auxMat[j] = {
                    'Bt': self.Bti[j],
                    'BBt': BBt,
                    'Spt': Spt,
                    'Sp': Sp,
                    'invSpt': Spt @ DSp,
                    'invSp': Sp @ DSp
                }

                self.auxMat[j]['BABt'] = sp.kron(Bxi[j]@sp.block_diag((Axi[j], Axi[j]))@Bxi[j].T, Mz_@Mz@Mz_.T) \
                     + sp.kron(Bxi[j]@sp.block_diag((Mxi[j], Mxi[j]))@Bxi[j].T, Mz_@Az@Mz_.T) \
                     + sp.kron(Mx_i[j]@Axi[j]@Mx_i[j].T, Bz@Mz@Bz.T) \
                     + sp.kron(Mx_i[j]@Mxi[j]@Mx_i[j].T, Bz@Az@Bz.T)

                self.auxMat[j]['Su0'] = self.A0i[j].assembly()
        
        self.Nu = Nu
        self.Np = Np
        self.coarse_dof = self.Ai[0].shape[0] + self.Bi[0].shape[0]
        case = 1

        if case == 0:
            self.bigAi = (self.Ai[0].assembly(), self.Bti[0].assembly(), self.Bi[0].assembly())

        # # ---- 构造 pressure nullspace ----
        # ns_vec = M.createVecRight()
        # ns_vec.set(0.0)

        # is_u, is_p = M.getNestISs()[1]
        # p_sub = ns_vec.getSubVector(is_p)
        # p_sub.set(1.0)
        # p_sub.assemble()
        # ns_vec.restoreSubVector(is_p, p_sub)

        # ns_vec.normalize()

        # nullspace = PETSc.NullSpace().create(vectors=[ns_vec])
        # M.setNullSpace(nullspace)
        # M.setNearNullSpace(nullspace)
        # # ---------------------------------

        # ksp = PETSc.KSP().create()
        # ksp.setOperators(M)
        # ksp.setType('preonly') 
        # ksp.getPC().setType('lu')

        # # ksp.setUp()
        # ksp.setFromOptions()
        # self.bigAi = ksp

        A = self.Ai[0].assembly().tocsr().astype(bm.float64)
        Bt = self.Bti[0].assembly().tocsr().astype(bm.float64)
        B = self.Bi[0].assembly().tocsr().astype(bm.float64)

        vel_dofs = bm.arange(A.shape[0], dtype=bm.int32)
        pres_dofs = bm.arange(A.shape[0], A.shape[0] + B.shape[0], dtype=bm.int32)

        A = csr_to_petsc_mat(A)
        Bt = csr_to_petsc_mat(Bt)
        B = csr_to_petsc_mat(B)

        np_ = B.getSize()[0]
        Z = PETSc.Mat().createAIJ([np_, np_])
        Z.setUp()
        Z.assemble()

        M = PETSc.Mat().createNest([[A, Bt],
                                    [B, Z]])
        M.assemble()

        velocity_is = PETSc.IS().createGeneral(vel_dofs)
        pressure_is = PETSc.IS().createGeneral(pres_dofs)

        # nullspace
        ns_vec = M.createVecRight()
        p_sub = ns_vec.getSubVector(pressure_is)
        p_sub.set(1.0)     
        p_sub.assemble()
        ns_vec.restoreSubVector(pressure_is, p_sub)
        ns_vec.normalize()

        nullspace = PETSc.NullSpace().create(vectors=(ns_vec,), constant=False)
        M.setNullSpace(nullspace)

        # # KSP
        # ksp = PETSc.KSP().create(comm=M.comm)
        # ksp.setOperators(M)
        # ksp.setType('fgmres')
        # ksp.setTolerances(rtol=1e-8, atol=1e-10, max_it=1000)

        # pc = ksp.getPC()
        # pc.setType('fieldsplit')
        # pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        # pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)

        # # 指定 field split
        # pc.setFieldSplitIS(("velocity", velocity_is), ("pressure", pressure_is))

        # ksp.setUp()
        # vel_pc = pc.getFieldSplitSubKSP()[0].getPC()
        # vel_pc.setType('hypre')
        # pres_pc = pc.getFieldSplitSubKSP()[1].getPC()
        # pres_pc.setType('jacobi')

        ksp = PETSc.KSP().create()
        ksp.setOperators(M)
        ksp.setType('preonly') 
        ksp.getPC().setType('lu')

        # ksp.setUp()
        ksp.setFromOptions()

        self.bigAi = ksp

    def vcycle(self, ru, rp, J=None):
        if J is None:
            J = self.level - 1
        if J == 0:
            start = time.time()
            r = bm.concat([ru, rp], axis=0)
            n = len(rp)
            e = self.solve(self.bigAi, r)
            self.coarse_count += 1
            self.coarse_time += time.time() - start
            ep = e[-n:]
            ep = ep - bm.mean(ep)
            return e[:-n], ep
        
        P_u = self.P_u[J-1]
        P_p = self.P_p[J-1] 
        R_u = self.R_u[J-1]
        R_p = self.R_p[J-1] 

        start = time.time()
        self.assembly_time += time.time() - start

        # pre-smoothing
        eu, ep = self.smoothing(bm.zeros((3*self.Nu[J],), dtype=bm.float64),
                                bm.zeros((self.Np[J],), dtype=bm.float64),ru,rp,J)
        if self.smoothing_times > 1:
            for _ in range(self.smoothing_times-1):
                eu, ep = self.smoothing(eu,ep,ru,rp,J)

        # form residual and restrict onto coarse grid
        start = time.time()
        rru = ru - self.Ai[J] @ eu - self.Bti[J] @ ep
        rrp = rp - self.Bi[J] @ eu

        ruc = R_u @ rru
        rpc = R_p @ rrp
        self.cycle_MUL_time += time.time() - start
        # coarse grid correction
        euc, epc = self.vcycle(ruc, rpc, J-1)

        # correction on the fine grid
        start = time.time()
        tempeu = P_u @ euc
        tempep = P_p @ epc
        self.cycle_MUL_time += time.time() - start
        eu += tempeu
        ep += tempep

        # post-smoothing
        for _ in range(self.smoothing_times):
            eu, ep = self.smoothing(eu,ep,ru,rp,J)
        # del self.auxMat[J]['Su0']
        return eu, ep   

    def smoothing(self, u, p, f, g, J):
        """Solve LUe = r.
        """
        auxMat = self.auxMat[J]
        smootherOpt = self.options
        A = self.A0i[J]
        B = self.Bi[J]
        start = time.time()
        smoother = StokesLSCDGS(auxMat,smootherOpt)
        u, p, self.SGS_time, self.MUL_time = smoother.run(u,p,f,g,A,B,self.SGS_time,self.MUL_time)
        t = time.time() - start
        print(t,'hh')
        self.smoothing_time += t
        self.smoothing_count += 1
        return u, p - bm.mean(p)   
    
    @variantmethod('direct')
    def solve(self, bigA, F, solver='mumps'):
        """
        Solve the linear system using direct method.
        """
        case = 1
        ksp = self.bigAi
        
        if case == 0:
            x = spsolve(ksp, F)
            return x

        from petsc4py import PETSc
        rhs = PETSc.Vec().createSeq(len(F))
        rhs.setArray(F)
        rhs.assemble()

        x = PETSc.Vec().createSeq(len(F))
        ksp.solve(rhs, x)

        return x.getArray()

    @solve.register('op')
    def solve(self, op: StokesOperator, F, solver='mumps'):
        """
        Solve the linear system using op method.
        """
        # from scipy.sparse.linalg import bicgstab, minres, gmres, cg, LinearOperator
        from fealpy.solver import bicgstab, minres, gmres, cg
        # linop = LinearOperator(shape=op.shape,matvec=op)
        # x, info = bicgstab(op, F)
        x, info = minres(op, F, atol=1e-8, rtol=1e-8)
        # x, info = cg(op, F, returninfo=True)
        print(info)
        return x

    @solve.register('mg')
    def solve(self, op: StokesOperator, F):
        # initial set up
        start = time.time()
        self.setup(op)
        self.setup_time += time.time() - start
        print(self.initial_assembly_time,self.setup_time)
        self.logger.info(f'Step 4. setup 完成\n')
        bigu = bm.zeros_like(F)
        bigr = F

        k = 0
        nb = bm.linalg.norm(F)
        err = bm.zeros((self.maxIt, 1), dtype=bm.float64)
        err[0] = bm.linalg.norm(bigr) / nb
        self.logger.info(f'Step 5. 进入主循环迭代\n')
        start = time.time()
        while (bm.max(err[k]) > self.tol) & (k <= self.maxIt):
            k = k + 1
            pdof = self.Np[-1]
            if self.cycle_type == 'VCYCLE':
                eu, ep = self.vcycle(bigr[:-pdof], bigr[-pdof:])
                # bigerru = self.vcycle(bigr)
            elif self.cycle_type == 'WCYCLE':
                eu, ep = self.wcycle(bigr[:-pdof], bigr[-pdof:])
                # bigerru = self.wcycle(bigr)
            
            bigerru = bm.concat([eu, ep])
            bigu = bigu + bigerru
            bigr = bigr - op @ bigerru

            # compute the relative error
            err[k] = bm.linalg.norm(bigr) / nb

            print(
                f'MG Vcycle iter: {k:2d},   '
                f'err = {bm.max(err[k, :]):8.4e}'
            )           

        self.auxMat
        err = err[:k]
        itStep = k
        cost = time.time() - start
        self.logger.info(f'Step 6. 程序结束, 开始输出打印结果\n')
        # Output
        print(f"iter: {itStep:2.0f},  "
            f"err = {max(err[-1]):8.4e},  "
            f"level = {self.level},   "
            f"total dof: {self.total_dof:2.0f},   "
            f"coarse dof: {self.coarse_dof:2.0f}\n\n"
            f"total time in coarsest grid: {self.coarse_time}\n"
            f"total time in SGS: {self.SGS_time}\n"
            f"total time in MUL of smoothing: {self.MUL_time}\n"
            f"total time in smoothing: {self.smoothing_time}\n"
            f"total time in cycle assembly: {self.assembly_time}\n"
            f"total time in cycle MUL: {self.cycle_MUL_time}\n"
            f"total time: {cost}\n\n"
            f"粗网格上求解次数: {self.coarse_count}\n"
            f"粗网格总时间占比: {self.coarse_time / cost},  \n"
            f"SGS平滑总时间占比: {self.SGS_time / cost},  \n"
            f"平滑@计算总时间占比: {self.MUL_time / cost},  \n"
            f"Smoothing总时间占比: {self.smoothing_time / cost},   \n"
            f"粗网格和平滑总时间占比: {(self.coarse_time+self.smoothing_time) / cost},   \n"
            f"A0组装时间占比: {self.assembly_time / cost},  \n"
            f"cycle矩阵乘法时间占比: {self.cycle_MUL_time / cost},  \n\n"
            f"矩阵初次组装时间: {self.initial_assembly_time},   \n"
            f"setup时间: {self.setup_time},   \n")
        
        if k > self.maxIt:
            print("NOTE: the iterative method does not converge!")

        return bigu

    @solve.register('amg')
    def solve(self, A, F):
        raise NotImplementedError("AMG solver not yet implemented.")

    def run(self):
        import time
        start = time.time()
        op0, A, F = self.linear_system()
        self.logger.info(f'Step 1. 完成初步线性系统组装\n')
        op, F1, BdDof = self.apply_bc(op0, bm.copy(F))
        F1[-self.n_p:] -= bm.mean(F1[-self.n_p:])
        del op0
        gc.collect()
        self.logger.info(f'Step 2. 完成边界自由度处理\n')

        import time
        start = time.time()
        self.solver = 'mg'
        self.solver = 'direct'
        if self.solver == 'direct':
            # BC = DirichletBC(
            #     (self.uspace, self.pspace),
            #     gd=(self.velocity_dirichlet, self.pressure_dirichlet),
            #     threshold=(self.is_velocity_boundary, self.is_pressure_boundary),
            #     method='interp'
            # )
            BC = DirichletBC(
                (self.uspace, self.pspace),
                gd=self.pde.velocity_dirichlet,
                threshold=self.pde.is_velocity_boundary,
                method='interp'
            )
            A, F2 = BC.apply(A, F)
            print(f'开始求解')
            tmr = timer()
            next(tmr)
            x = spsolve(A.to_scipy(), F2)
            # x = self.solve['direct'](op, F1)
            tmr.send(f'求解器时间')
            next(tmr)
        
        elif self.solver == 'mg':           
            bd_flag = bm.zeros((len(F),), dtype=bm.bool)
            bm.set_at(bd_flag, BdDof, True)
            self.logger.info(f'Step 3. 开始多重网格setup阶段\n')
            self.initial_assembly_time += time.time() - start
            x_in = self.solve['mg'](op, F1[~bd_flag])
            x = bm.set_at(F1, ~bd_flag, x_in)
        
        self.uh[:] = x[:3*self.ugdof]
        ph = x[3*self.ugdof:]
        print(x[:3*self.ugdof].max(),x[:3*self.ugdof].min())
        # self.post_process(uh ,ph)
        print(f'error: {self.error()}')
        return self.error()
    
    def error(self):
        l2 = self.mesh.error(self.pde.velocity, self.uh)
        return l2
    
    def post_process(self, uh, ph):
        iNN = self.imesh.number_of_nodes()
        tNN = self.tmesh.number_of_nodes()
        tgdof = self.tmesh.number_of_global_ipoints(p=2)
        igdof = self.imesh.number_of_global_ipoints(p=2)
        gdof = tgdof * igdof
        idx = bm.arange(gdof).reshape(tgdof, -1)[:tNN, :iNN].ravel()

        self.mesh.nodedata['ph'] = ph
        self.mesh.nodedata['uh'] = uh.reshape(3,-1).T[idx,:]
        
        self.mesh.to_vtk('dld_prism_chip.vtu')
