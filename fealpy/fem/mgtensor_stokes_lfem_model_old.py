      
from typing import Optional, Union
from fealpy.backend import bm

from fealpy.typing import TensorLike
from fealpy.decorator import variantmethod, cartesian

from fealpy.mesh import TriangleMesh, IntervalMesh
from fealpy.functionspace import LagrangeFESpace, functionspace
from fealpy.fem import BilinearForm, LinearForm, DirichletBC, BlockForm, LinearBlockForm
from fealpy.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator, PressWorkIntegrator, CouplingMassIntegrator
from fealpy.model import PDEModelManager, ComputationalModel

from fealpy.mesher import DLDMicrofluidicChipMesher

from fealpy.sparse import spdiags, coo_matrix, csr_matrix, COOTensor, CSRTensor
from fealpy.solver import cg, spsolve, transferP1red, transferP2red, StokesLSCDGS, indofP1, indofP2
from fealpy.utils import timer

import scipy.sparse as sp
import scipy.sparse.linalg as lg
import time

import tracemalloc

"""
1. 减小矩阵规模来作用内部自由度, 相应的减少插值、限制矩阵规模
2. 整体方案
    Plan I: 全体使用算子, 结合kron积快速平滑 (最优解, 但需要积累)
    Plan II: 全体使用算子, 不进行快速平滑 (GPU下, 目前最优解)
    Plan III: 除去平滑, 使用Operator (居中方案)

"""

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
    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.m0, self.n0 = A.shape
        self.m1, self.n1 = B.shape
        self.shape = (self.m0*self.m1, self.n0*self.n1)

    def __matmul__(self, x):
        X = bm.reshape(x, (self.n0, self.m1))
        Y = self.A.to_scipy() @ X @ self.B.to_scipy()
        Y = Y.ravel()
        return Y


class StokesOperator(LinearOperator):
    def __init__(self, Ax, Mx, Az, Mz, Bx, Bz, Mx_, Mz_):
        self.Ax = Ax.assembly().to_scipy()
        self.Mx = Mx.assembly().to_scipy()
        self.Az = Az.assembly().to_scipy()
        self.Mz = Mz.assembly().to_scipy()
        self.Bx = Bx.assembly().to_scipy().T
        self.Bz = Bz.assembly().to_scipy().T
        self.Mx_ = Mx_.assembly().to_scipy().T
        self.Mz_ = Mz_.assembly().to_scipy().T
        self.inflag_uz = None
        self.inflag_pz = None
        self.inflag_u = None
        self.inflag_p = None
        self.BdDof = None
        self.fix = False
        self.res_mat = False
        self.set_up()

    def set_up(self):
        if self.inflag_u is not None:
            inflag_u = bm.to_numpy(self.inflag_u)
            Biginflag_u = bm.to_numpy(bm.concat([self.inflag_u,self.inflag_u], axis=0))
            inflag_uz = bm.to_numpy(self.inflag_uz)
            self.fix = False
            self.res_mat = True
            self.Ax = self.Ax[inflag_u][:,inflag_u]
            self.Mx = self.Mx[inflag_u][:,inflag_u]
            self.Az = self.Az[inflag_uz][:,inflag_uz]
            self.Mz = self.Mz[inflag_uz][:,inflag_uz]
            if self.inflag_p is not None:
                inflag_p = bm.to_numpy(self.inflag_p)
                self.Bx = self.Bx[inflag_p][:,Biginflag_u]
                self.Mx_ = self.Mx_[inflag_p][:,inflag_u]
                self.Mz_ = self.Mz_[:,inflag_uz]
                self.Bz = self.Bz[:,inflag_uz]
                # self.Mz_ = self.Mz_[inflag_pz][:,inflag_uz]
                # self.Bz = self.Bz[inflag_pz][:,inflag_uz]
            else:
                self.Bx = self.Bx[:,Biginflag_u]
                self.Mx_ = self.Mx_[:,inflag_u]
                self.Mz_ = self.Mz_[:,inflag_uz]
                self.Bz = self.Bz[:,inflag_uz]
                # self.Mz_ = self.Mz_[:,inflag_uz]
                # self.Bz = self.Bz[:,inflag_uz]

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

    def __matmul__(self, x):
        v = bm.copy(x)
        if self.fix:
            val = v[self.BdDof]
            v = bm.set_at(v, self.BdDof, 0.0)

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
        if self.fix:
            bm.set_at(y, self.BdDof, val) 
        elif not self.res_mat:
            self.fix = True
        return y


class MGTensorStokesLFEMModelOLD(ComputationalModel):
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

    def set_init_mesher(self, mesher: DLDMicrofluidicChipMesher, imesh: IntervalMesh):
        """
        Set the initial mesh for the simulation.
        
        Parameters:
            mesh: The computational mesh object
        """
        tmesh = mesher.mesh
        self.mesh0 = TriangleMesh(tmesh.entity('node'), tmesh.entity('cell'))
        self.mesh1 = TriangleMesh(tmesh.entity('node'), tmesh.entity('cell'))
        tmesh.uniform_refine(self.level-1)
        self.tmesh = tmesh
        
        # import matplotlib.pyplot as plt
        # from fealpy.mesh import TensorPrismMesh
        # # mesh = TensorPrismMesh(self.tmesh, imesh)
        # mesh = tmesh
        # ipoints = tmesh.interpolation_points(p=1)
        # fig = plt.figure()
        # axes = fig.add_subplot(111)
        # mesh.add_plot(axes)
        # mesh.find_node(axes, node=ipoints, 
        #             showindex=True, color='r', fontsize='10')
        # # tmesh.find_cell(axes, showindex=True, fontsize='35')
        # plt.show()

        self.radius = mesher.radius
        self.centers = mesher.centers
        self.boundary = mesher.boundary
        self.inlet_boundary = mesher.inlet_boundary
        self.outlet_boundary = mesher.outlet_boundary
        self.wall_boundary = mesher.wall_boundary
        self.project_edges = mesher.project_edges
        self.imesh = imesh

    def set_space_degree(self, p: int=2):
        """
        Set the polynomial degree for function spaces
        """
        self.p = p

    def set_inlet_condition(self)-> None:
        """
        Set the PDE data for the model.
        """
        @cartesian
        def inlet_velocity(p: TensorLike) -> TensorLike:
            """Compute exact solution of velocity."""
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
            result = bm.zeros(p.shape, dtype=bm.float64)
            result[..., 0] = 10**2 *y * (1-y) * z * (0.1-z)
            result[..., 1] = bm.array(0.0)
            return result
        
        @cartesian
        def wall_velocity(p: TensorLike) -> TensorLike:
            """Compute exact solution of velocity."""
            x = p[..., 0]
            y = p[..., 1]
            result = bm.zeros(p.shape, dtype=bm.float64)
            result[..., 0] = bm.array(0.0)
            result[..., 1] = bm.array(0.0)
            return result
        
        @cartesian
        def obstacle_velocity(p: TensorLike) -> TensorLike:
            """Compute exact solution of velocity."""
            x = p[..., 0]
            y = p[..., 1]
            result = bm.zeros(p.shape, dtype=bm.float64)
            result[..., 0] = bm.array(0.0)
            result[..., 1] = bm.array(0.0)
            return result
        
        @cartesian
        def outlet_pressure(p: TensorLike) -> TensorLike:
            """Compute exact solution of velocity."""
            x = p[..., 0]
            y = p[..., 1]
            result = bm.zeros(p.shape[0], dtype=bm.float64)
            result[:] = 0.0
            return result
        
        @cartesian
        def is_inlet_boundary( p: TensorLike) -> TensorLike:
            """Check if point where velocity is defined is on boundary."""
            bd = self.inlet_boundary
            return self.is_lateral_boundary(p, bd)
        
        @cartesian
        def is_outlet_boundary( p: TensorLike) -> TensorLike:
            """Check if point where pressure is defined is on boundary."""
            bd = self.outlet_boundary
            return self.is_lateral_boundary(p, bd)

        @cartesian
        def is_wall_boundary(p: TensorLike) -> TensorLike:
            """Check if point where velocity is defined is on boundary."""
            bd = self.wall_boundary
            return self.is_lateral_boundary(p, bd)
        
        @cartesian
        def is_top_or_bottom(p: TensorLike) -> TensorLike:
            """Check if point where velocity is defined is on top or bottom boundary."""
            atol = 1e-12
            thickness = self.thickness
            cond = (bm.abs(p[:, -1]) < atol) | (bm.abs(p[:, -1] - thickness) < atol)
            return cond
        
        @cartesian
        def is_obstacle_boundary(p: TensorLike) -> TensorLike:
            """Check if point where velocity is defined is on boundary."""
            x = p[..., 0]
            y = p[..., 1]
            radius = self.options['radius']
            atol = 1e-12
            on_boundary = bm.zeros_like(x, dtype=bool)
            for center in self.centers:
                cx, cy = center
                on_boundary |= (x - cx)**2 + (y - cy)**2 < radius**2 + atol
            return on_boundary
        
        self.inlet_velocity = inlet_velocity
        self.wall_velocity = wall_velocity
        self.obstacle_velocity = obstacle_velocity
        self.outlet_pressure = outlet_pressure

        self.is_inlet_boundary = is_inlet_boundary
        self.is_outlet_boundary = is_outlet_boundary
        self.is_wall_boundary = is_wall_boundary
        self.is_top_or_bottom = is_top_or_bottom
        self.is_obstacle_boundary = is_obstacle_boundary

    def is_lateral_boundary(self, p: TensorLike, bd: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        atol = 1e-12
        v0 = p[:, None, :-1] - bd[None, 0::2, :] # (NN, NI, 2)
        v1 = p[:, None, :-1] - bd[None, 1::2, :] # (NN, NI, 2)

        cross = v0[..., 0]*v1[..., 1] - v0[..., 1]*v1[..., 0] # (NN, NI)
        dot = bm.einsum('ijk,ijk->ij', v0, v1) # (NN, NI)
        cond = (bm.abs(cross) < atol) & (dot < atol)
        return bm.any(cond, axis=1)
    
    @cartesian
    def is_velocity_boundary(self, p: TensorLike, dim=3) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        inlet = self.is_inlet_boundary(p)
        wall = self.is_wall_boundary(p)
        top_or_bottom = self.is_top_or_bottom(p)
        obstacle = self.is_obstacle_boundary(p)
        if dim == 2:
            return inlet | wall | obstacle
        return inlet | wall | top_or_bottom | obstacle
    
    @cartesian
    def is_pressure_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        return self.is_outlet_boundary(p)

    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed velocity on boundary, if needed explicitly."""
        inlet = self.inlet_velocity(p)
        is_inlet = self.is_inlet_boundary(p)
        
        result = bm.zeros_like(p, dtype=p.dtype)
        result[is_inlet] = inlet[is_inlet]

        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed pressure on boundary (usually for stability)."""
        outlet = self.outlet_pressure(p)
        is_outlet = self.is_outlet_boundary(p)
        result = bm.zeros_like(p[..., 0], dtype=p.dtype)
        result[is_outlet] = outlet[is_outlet]
        return result

    @variantmethod
    def linear_system(self):
        """
        Assemble the linear system for the Stokes equations.
        """
        # from fealpy.mesh import TensorPrismMesh
        # self.mesh = TensorPrismMesh(self.tmesh, self.imesh)
        
        # self.uspace = functionspace(self.mesh, ('Lagrange', 2), shape=(3, -1))
        # self.pspace = functionspace(self.mesh, ('Lagrange', 1))
        
        self.int_space0 = LagrangeFESpace(self.imesh, p=1)
        self.int_space1 = LagrangeFESpace(self.imesh, p=2)
        self.tri_space0 = LagrangeFESpace(self.tmesh, p=1)
        self.tri_space1 = LagrangeFESpace(self.tmesh, p=2)

        Ax = BilinearForm(self.tri_space1)
        Ax.add_integrator(ScalarDiffusionIntegrator())

        Mx = BilinearForm(self.tri_space1)
        Mx.add_integrator(ScalarMassIntegrator())

        Az = BilinearForm(self.int_space1)
        Az.add_integrator(ScalarDiffusionIntegrator())

        Mz = BilinearForm(self.int_space1)
        Mz.add_integrator(ScalarMassIntegrator())

        self.uspace2d = functionspace(self.tmesh, ('Lagrange', 2), shape=(2, -1))
        self.pspace2d = functionspace(self.tmesh, ('Lagrange', 1))

        Bx = BilinearForm((self.pspace2d, self.uspace2d))
        self.BPx = PressWorkIntegrator()
        self.BPx.coef = -1.0
        Bx.add_integrator(self.BPx)

        Mz_ = BilinearForm((self.int_space0, self.int_space1))
        Mz_.add_integrator(CouplingMassIntegrator())
 
        self.uspace1d = functionspace(self.imesh, ('Lagrange', 2), shape=(1, -1))
        self.pspace1d = functionspace(self.imesh, ('Lagrange', 1))

        Bz = BilinearForm((self.pspace1d, self.uspace1d))
        self.BPz = PressWorkIntegrator()
        self.BPz.coef = -1.0
        Bz.add_integrator(self.BPz)
        
        Mx_ = BilinearForm((self.tri_space0, self.tri_space1))
        Mx_.add_integrator(CouplingMassIntegrator())
        self.ugdof = Ax.shape[0]*Mz.shape[0]
        print(f'自由度个数: {Ax.shape[0]*Mz.shape[0]*3+Bx.shape[1]*Mz_.shape[1]}')
        import ipdb;ipdb.set_trace()
        stokes_operator = StokesOperator(Ax, Mx, Az, Mz, Bx, Bz, Mx_, Mz_)
       
        # A1 = sp.kron(Ax.assembly().to_scipy(), Mz.assembly().to_scipy()) + \
        #      sp.kron(Mx.assembly().to_scipy(), Az.assembly().to_scipy())
        # B0 = sp.kron(Bx.assembly().to_scipy().T, Mz_.assembly().to_scipy().T)
        # B1 = sp.kron(Mx_.assembly().to_scipy().T, Bz.assembly().to_scipy().T)
        
        # A0 = sp.block_diag((A1, A1, A1))
        # B = sp.bmat([[B0, B1]])
        # A = sp.bmat([[A0, B.T],
        #              [B, None]])

        # from fealpy.sparse import COOTensor
        # A = COOTensor(
        #     indices=bm.stack([A.row, A.col], axis=0),
        #     values=A.data,
        #     spshape=A.shape
        # )
        A = None
        self.n_A = stokes_operator.n_A
        self.n_p = stokes_operator.n_p
        self.x0 = bm.zeros((self.n_A,), dtype=bm.float64)
        F = bm.zeros((self.n_A,), dtype=bm.float64)
        return stokes_operator, A, F
    
    def boundary_dof_index(self):
        isDDof0 = self.tmesh.boundary_node_flag()
        isDDof1 = self.tri_space1.is_boundary_dof()
        isDDof2 = self.imesh.boundary_face_flag()
        igdof = self.int_space1.number_of_global_dofs()
        isDDof3 = bm.zeros((igdof, ), dtype=bm.bool)
        bm.set_at(isDDof3, bm.arange(len(isDDof2)), isDDof2)
        # isDDof3 = self.int_space1.is_boundary_dof()

        bd_dof0 = ~((~isDDof1[:, None]) * (~isDDof3[None, :])).ravel()
        bd_dof1 = ~((~isDDof0[:, None]) * (~isDDof2[None, :])).ravel()

        return (bd_dof1, bd_dof0)

    def interpolation_points(self):
        ipoint0 = self.imesh.interpolation_points(p=1)
        ipoint1 = self.imesh.interpolation_points(p=2)
        ipoint2 = self.tmesh.interpolation_points(p=1)
        ipoint3 = self.tmesh.interpolation_points(p=2)
        
        p0 = bm.concat([bm.repeat(ipoint2, ipoint0.shape[0], axis=0), 
                          bm.tile(ipoint0.T, (ipoint2.shape[0],)).T], axis=1)
        p1 = bm.concat([bm.repeat(ipoint3, ipoint1.shape[0], axis=0), 
                          bm.tile(ipoint1.T, (ipoint3.shape[0],)).T], axis=1)
        
        return (p0, p1)

    def apply_bc(self, stokes_operator: StokesOperator, F):
        uh = self.x0
        gd = (self.velocity_dirichlet, self.pressure_dirichlet)
        threshold = (self.is_velocity_boundary, self.is_pressure_boundary)
        
        dofs = self.boundary_dof_index()
        points = self.interpolation_points() # (2000w, 3) ~ 500 MB
        basic = [3*len(points[1]), 0]
        BdDof = []

        for i in range(2):
            index_dof = bm.arange(len(points[i]))[dofs[i]] + basic[i]
            # ipoints: (NI， 3), 边界插值点坐标, 
            bd_point = points[i][dofs[i]] 
            # flag: (NI,), 判断边界点是否属于某类边界
            flag = threshold[1-i](bd_point)
            index_dof = index_dof[flag]
            val = gd[1-i](bd_point[flag])
            if i == 1:
                index_dof = bm.concat([index_dof, index_dof + len(points[1]), 
                                    index_dof + 2*len(points[1])], axis=0)
                # import ipdb;ipdb.set_trace()
                val = val.T.reshape(-1)

            BdDof.append(index_dof)
            isBdDof = bm.zeros(self.n_A, dtype=bm.bool)
            isBdDof = bm.set_at(isBdDof, index_dof, True)
            uh = bm.set_at(uh, (..., isBdDof), val)

        BdDof = bm.concat([BdDof[1], BdDof[0]], axis=0)
        F = F - stokes_operator @ uh # 5000w ~ 400MB
        F = bm.set_at(F, BdDof, uh[BdDof])

        return BdDof, F

    def from_scipy(self, M):
        row = bm.from_numpy(M.row)
        col = bm.from_numpy(M.col)
        data = bm.from_numpy(M.data)
        M = COOTensor(indices=bm.stack([row, col], axis=0),
                        values=data, spshape=M.shape)
        
        return M

    def setup(self, op: StokesOperator):
        """Compute restriction and interpolation operators.
        """
        flag = self.imesh.boundary_face_flag()
        igdof = self.int_space1.number_of_global_dofs()
        isDDof = bm.zeros((igdof, ), dtype=bm.bool)
        bm.set_at(isDDof, bm.arange(len(flag)), flag)
        op.inflag_pz = ~flag
        op.inflag_uz = ~isDDof
        op.inflag_u = indofP2(self.tmesh, threshold=self.is_velocity_boundary)
        op.inflag_p = indofP1(self.tmesh, threshold=self.is_pressure_boundary)
        op.set_up()
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
        Nu[-1] = op.n_Ax
        Np[-1] = op.n_Bx
        
        # Compute Pro and Res of u and p.
        Pro_p = transferP1red(self.mesh0, self.level, self.is_pressure_boundary)
        Pro_u = transferP2red(self.mesh1, self.level, self.is_velocity_boundary)
        
        for j in range(level - 1, 0, -1):
            Axi[j-1] = Pro_u[j-1].T @ Axi[j] @ Pro_u[j-1]
            Mxi[j-1] = Pro_u[j-1].T @ Mxi[j] @ Pro_u[j-1]
            Bxi[j-1] = Pro_p[j-1].T @ Bxi[j] @ sp.block_diag([Pro_u[j-1],Pro_u[j-1]])
            Mx_i[j-1] = Pro_p[j-1].T @ Mx_i[j] @ Pro_u[j-1]

        P_u = [None] * (level-1)
        P_p = [None] * (level-1)
        R_u = [None] * (level-1)
        R_p = [None] * (level-1)

        auxMat = [None] * level
        Ai = [None] * level
        Bi = [None] * level
        bigAi = [None] * level

        Iz2 = spdiags(bm.ones((op.n_Mz,)), 0, op.n_Mz, op.n_Mz)
        Iz1 = spdiags(bm.ones((op.n_Mz_,)), 0, op.n_Mz_, op.n_Mz_)

        for j in range(self.level):
            A0 = sp.kron(Axi[j], Mz) + sp.kron(Mxi[j], Az)
            Ai[j] = sp.block_diag([A0, A0, A0])
            Bi[j] = sp.bmat([[sp.kron(Bxi[j], Mz_), sp.kron(Mx_i[j], Bz)]])
            Nu[j] = A0.shape[0]
            Np[j] = Bi[j].shape[0]
            # bigAi[j] = (sp.bmat([[Ai[j], Bi[j].T],[Bi[j], None]]).tocsr())
            if j < self.level - 1:
                P0 = sp.kron(Pro_u[j], Iz2.to_scipy())
                P_u[j] = self.from_scipy(sp.block_diag([P0, P0, P0]))
                P_p[j] = self.from_scipy(sp.kron(Pro_p[j], Iz1.to_scipy()))
            
            if j > 0:
                Bt = Bi[j].T
                BBt = Bi[j] @ Bt
                BABt = Bi[j] @ Ai[j] @ Bt

                Su = sp.tril(A0)
                Sp = sp.tril(BBt)
                Spt = sp.triu(BBt)
                DSp = BBt.diagonal()
                
                Bt = self.from_scipy(Bt)
                BBt = self.from_scipy(BBt.tocoo()).tocsr()
                BABt = self.from_scipy(BABt.tocoo()).tocsr()

                Spt = self.from_scipy(Spt)
                Sp = self.from_scipy(Sp)
                DSp = bm.tensor(DSp)

                auxMat[j] = {
                    'Bt': Bt,
                    'BBt': BBt,
                    'BABt': BABt,
                    'Su': Su.tocsr(),
                    'Spt': Spt.to_scipy().tocsr(),
                    'Sp': Sp.to_scipy().tocsr(),
                    'DSp': DSp
                }
            Ai[j] = self.from_scipy(Ai[j])
            Bi[j] = self.from_scipy(Bi[j])

        self.P_u = P_u
        self.P_p = P_p
        
        self.Ai = Ai
        self.Bi = Bi
        self.Nu = Nu
        self.Np = Np
        self.auxMat = auxMat

    def vcycle(self, r, J=None):
        if J is None:
            J = self.level - 1
        if J == 0:
            import pyamg
            start = time.time()
            # mg = pyamg.ruge_stuben_solver(self.bigAi[J])
            # e = mg.solve(r)
            from fealpy.sparse.ops import bmat
            bigAi = (bmat([[self.Ai[0].tocsr(), self.Bi[0].T.tocsr()],[self.Bi[0].tocsr(), None]]))
            e = spsolve(bigAi, r)
            self.coarse_count += 1
            self.coarse_time += time.time() - start
            self.bigAi = bigAi
            return e
        
        P_u = self.P_u[J-1]
        P_p = self.P_p[J-1]
        
        ru = r[:3*self.Nu[J]]
        rp = r[3*self.Nu[J]:]
        
        # pre-smoothing
        eu, ep = self.smoothing(bm.zeros((3*self.Nu[J],), dtype=bm.float64),
                                bm.zeros((self.Np[J],), dtype=bm.float64),ru,rp,J)
        if self.smoothing_times > 1:
            for _ in range(self.smoothing_times-1):
                eu, ep = self.smoothing(eu,ep,ru,rp,J)

        # form residual and restrict onto coarse grid
        rru = ru - self.Ai[J] @ eu - self.Bi[J].T @ ep
        rrp = rp - self.Bi[J] @ eu

        ruc = P_u.T @ rru
        rpc = P_p.T @ rrp
        # coarse grid correction
        rc = bm.concat([ruc, rpc], axis=0)
        ec = self.vcycle(rc, J-1)

        # correction on the fine grid
        tempeu = P_u @ ec[:3*self.Nu[J-1]]
        tempep = P_p @ ec[3*self.Nu[J-1]:]
        eu = tempeu + eu
        ep = tempep + ep

        # post-smoothing
        # eu, ep = self.smoothing(eu,ep,ru,rp,J)
        for _ in range(self.smoothing_times):
            eu, ep = self.smoothing(eu,ep,ru,rp,J)
        e = bm.concat([eu, ep], axis=0)
        return e       

    def wcycle(self, r, J=None): 
        if J is None:
            J = self.level - 1
        if J == 0:
            e = bm.zeros_like(r)
            start = time.time()
            # e[:-1] = lg.spsolve(self.bigAi[J].tocsr()[:-1, :-1], r[:-1])
            e = lg.spsolve(self.bigAi[J].tocsr(), r)
            self.coarse_time += time.time() - start
            self.coarse_count += 1
            return e
        
        Res_u = self.Pro_u[J-1].T
        Res_p = self.Pro_p[J-1].T
        
        ru = r[:2*self.Nu[J]]
        rp = r[2*self.Nu[J]:]
        
        # pre-smoothing
        eu, ep = self.smoothing(bm.zeros((2*self.Nu[J],)),bm.zeros((self.Np[J],)),ru,rp,J)
        if self.smoothing_times == 2:
            eu, ep = self.smoothing(eu,ep,ru,rp,J)

        # form residual and restrict onto coarse grid
        rru = ru - self.Ai[J] @ eu - self.Bi[J].T @ ep
        rrp = rp - self.Bi[J] @ eu

        ruc = (Res_u @ rru.reshape(2, -1).T).reshape(-1, order='F')
        rpc = Res_p @ rrp
        
        # coarse grid correction
        rc = bm.concat([ruc, rpc], axis=0)
        ec = self.wcycle(rc, J-1)
        # once more for w-cycle
        ec = ec + self.wcycle(rc - self.bigAi[J-1] @ ec,J-1)

        # correction on the fine grid
        tempeu = (self.Pro_u[J-1] @ (ec[:2*self.Nu[J-1]].reshape(2, -1).T)).reshape(-1, order='F')
        tempep = self.Pro_p[J-1] @ ec[2*self.Nu[J-1]:]
        eu = tempeu + eu
        ep = tempep + ep

        # post-smoothing
        eu, ep = self.smoothing(eu,ep,ru,rp,J)
        if self.smoothing_times == 2:
            eu, ep = self.smoothing(eu,ep,ru,rp,J)
        e = bm.concat([eu, ep], axis=0)
        return e       
    
    def smoothing(self, u, p, f, g, J):
        """Solve LUe = r.
        """
        auxMat = self.auxMat[J]
        smootherOpt = self.options
        A = self.Ai[J]
        B = self.Bi[J]
        start = time.time()
        smoother = StokesLSCDGS(auxMat,smootherOpt)
        u, p, self.SGS_time, self.MUL_time = smoother.run(u,p,f,g,A,B,self.SGS_time,self.MUL_time)
        t = time.time() - start
        print(t)
        self.smoothing_time += t
        self.smoothing_count += 1
        return u, p    
    
    @variantmethod('direct')
    def solve(self, stokes_operator: StokesOperator, F, solver='mumps'):
        """
        Solve the linear system using direct method.
        """
        # from scipy.sparse.linalg import bicgstab, minres, gmres, cg, LinearOperator
        from fealpy.solver import bicgstab, minres, gmres, cg
        # linop = LinearOperator(shape=stokes_operator.shape,matvec=stokes_operator)
        # x, info = bicgstab(stokes_operator, F)
        x, info = minres(stokes_operator, F, atol=1e-8, rtol=1e-8)
        # x, info = cg(stokes_operator, F, returninfo=True)
        print(info)
        return x

    @solve.register('mg')
    def solve(self, stokes_operator: StokesOperator, F):
        # initial set up
        
        self.setup(stokes_operator)
        self.logger.info(f'Step 4. setup 完成\n')
        bigF = F
        bigu = bm.zeros_like(F)
 
        bigr = bigF - stokes_operator @ bigu

        k = 0
        nb = bm.linalg.norm(bigF)
        err = bm.zeros((self.maxIt, 1), dtype=bm.float64)
        err[0] = bm.linalg.norm(bigr) / nb
        self.logger.info(f'Step 5. 进入主循环迭代\n')
        start = time.time()
        while (bm.max(err[k]) > self.tol) & (k <= self.maxIt):
            k = k + 1
            if self.cycle_type == 'VCYCLE':
                bigerru = self.vcycle(bigr)
            elif self.cycle_type == 'WCYCLE':
                bigerru = self.wcycle(bigr)
            bigu = bigu + bigerru
            bigr = bigr - stokes_operator @ bigerru

            # compute the relative error
            err[k] = bm.linalg.norm(bigr) / nb

            print(
                f'MG Vcycle iter: {k:2d},   '
                f'err = {bm.max(err[k, :]):8.4e}'
            )
        err = err[:k]
        itStep = k
        cost = time.time() - start
        self.logger.info(f'Step 6. 程序结束, 开始输出打印结果\n')
        # Output
        print(f"iter: {itStep:2.0f},  "
            f"err = {max(err[-1]):8.4e},  "
            f"level = {self.level},   "
            f"total dof: {self.bigAi[-1].shape[0]:2.0f}"
            f"coarse dof: {self.bigAi.shape[0]:2.0f}\n\n"
            f"total time in coarsest grid: {self.coarse_time}\n"
            f"total time in SGS: {self.SGS_time}\n"
            f"total time in MUL of smoothing: {self.MUL_time}\n"
            f"total time in smoothing: {self.smoothing_time}\n"
            f"total time: {cost}\n\n"
            f"粗网格上求解次数: {self.coarse_count}\n"
            f"粗网格总时间占比: {self.coarse_time / cost},  \n"
            f"SGS平滑总时间占比: {self.SGS_time / cost},  \n"
            f"平滑@计算总时间占比: {self.MUL_time / cost},  \n"
            f"Smoothing总时间占比: {self.smoothing_time / cost},   \n"
            f"粗网格和平滑总时间占比: {(self.coarse_time+self.smoothing_time) / cost}")
        
        if k > self.maxIt:
            print("NOTE: the iterative method does not converge!")

        return bigu

    @solve.register('amg')
    def solve(self, A, F):
        raise NotImplementedError("AMG solver not yet implemented.")

    def run(self):
        stokes_operator, A, F = self.linear_system()
        self.logger.info(f'Step 1. 完成初步线性系统组装\n')
        BdDof, F1 = self.apply_bc(stokes_operator, bm.copy(F))
        
        # summary.print_(summary.summarize(muppy.get_objects()))
        self.logger.info(f'Step 2. 完成边界自由度处理\n')
        import gc
        gc.collect()

        import time
        start = time.time()
        self.solver = 'mg'
        
        if self.solver == 'direct':
            BC = DirichletBC(
                (self.uspace, self.pspace),
                gd=(self.velocity_dirichlet, self.pressure_dirichlet),
                threshold=(self.is_velocity_boundary, self.is_pressure_boundary),
                method='interp'
            )
            A, F2 = BC.apply(A, F)
            print(f'开始求解')
            tmr = timer()
            next(tmr)
            x = spsolve(A.to_scipy(), F2)
            # x = self.solve['direct'](stokes_operator, F1)
            tmr.send(f'求解器时间')
            next(tmr)
            
        elif self.solver == 'mg':
            bd_flag = bm.zeros((len(F),), dtype=bm.bool)
            bm.set_at(bd_flag, BdDof, True)
            self.logger.info(f'Step 3. 开始多重网格setup阶段\n')
            x_in = self.solve['mg'](stokes_operator, F1[~bd_flag])
            x = bm.set_at(F1, ~bd_flag, x_in)
        
        uh = x[:3*self.ugdof]
        ph = x[3*self.ugdof:]
        print(ph.max(),uh.max())
        # self.post_process(uh ,ph)
        return uh, ph
    
    def error(self):
        err = bm.sqrt(bm.mean((self.pde.solution(self.node) - self.uh)**2))
        return err
    
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


    