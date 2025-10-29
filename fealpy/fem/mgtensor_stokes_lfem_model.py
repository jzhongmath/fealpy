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

from fealpy.sparse import spdiags, coo_matrix, csr_matrix
from fealpy.solver import cg, spsolve, StokesLSCDGS
from fealpy.utils import timer

import scipy.sparse as sp

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

        self.is_boundary_dof = None
        self.fix = False

    def __matmul__(self, x):
        if self.fix:
            val = x[self.is_boundary_dof]
            bm.set_at(x, self.is_boundary_dof, 0.0)

        U1 = bm.reshape(x[:self.n_u0], (self.n_Ax, self.n_Mz))
        U2 = bm.reshape(x[self.n_u0:2*self.n_u0], (self.n_Ax, self.n_Mz))
        U3 = bm.reshape(x[2*self.n_u0:3*self.n_u0], (self.n_Ax, self.n_Mz))

        U4 = bm.reshape(x[:2*self.n_u0], (self.m_Bx, self.m_Mz_))
        U5 = bm.reshape(x[2*self.n_u0:3*self.n_u0], (self.m_Mx_, self.m_Bz))

        P = bm.reshape(x[-self.n_p:], (self.n_Bx, self.n_Mz_))

        AU1 = self.Ax @ U1 @ self.Mz + self.Mx @ U1 @ self.Az
        AU2 = self.Ax @ U2 @ self.Mz + self.Mx @ U2 @ self.Az
        AU3 = self.Ax @ U3 @ self.Mz + self.Mx @ U3 @ self.Az

        BP1 = self.Bx.T @ P @ self.Mz_
        BP2 = self.Mx_.T @ P @ self.Bz
        
        BU1 = self.Bx @ U4 @ self.Mz_.T
        BU2 = self.Mx_ @ U5 @ self.Bz.T
        # import ipdb;ipdb.set_trace()
        l1 = bm.concat([AU1.ravel(), AU2.ravel()], axis=0) + BP1.ravel()
        l2 = AU3.ravel() + BP2.ravel()
        l3 = BU1.ravel() + BU2.ravel()

        y = bm.concat([l1, l2, l3], axis=0)
        if self.fix:
            bm.set_at(y, self.is_boundary_dof, val)
        else:
            self.fix = True
        return y


class MGTensorStokesLFEMModel(ComputationalModel):
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
            
        self.level = options.get('level')

        self.options = options
        self.x0 = options.get('x0', None)
        self.tol = options.get('tol', 1e-8)  
        self.maxIt = options.get('solvermaxit', 200)  
        self.N0 = options.get('N0', 500)
        self.mu = options.get('smoothingstep', 1)
        self.solver = options.get('solver', 'VCYCLE')

        self.preconditioner = options.get('preconditioner', 'none')
        self.coarsegridsolver = options.get('coarsegridsolver', 'direct')

        self.thickness = options.get('thickness', 0.1)

    def set_init_mesher(self, mesher: DLDMicrofluidicChipMesher, imesh: IntervalMesh):
        """
        Set the initial mesh for the simulation.
        
        Parameters:
            mesh: The computational mesh object
        """
        self.tmesh = mesher.mesh
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
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        inlet = self.is_inlet_boundary(p)
        wall = self.is_wall_boundary(p)
        top_or_bottom = self.is_top_or_bottom(p)
        obstacle = self.is_obstacle_boundary(p)
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
        
        print(Ax.shape[0]*Mz.shape[0]*3+Bx.shape[1]*Mz_.shape[1])
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
        self.n_A = stokes_operator.n_A
        self.n_u0 = stokes_operator.n_u0
        self.x0 = bm.zeros((self.n_A,), dtype=bm.float64)
        F = bm.zeros((self.n_A,), dtype=bm.float64)
        # print(self.n_A)
        return stokes_operator, F
    
    def boundary_dof_index(self):
        isDDof0 = self.tri_space0.is_boundary_dof()
        isDDof1 = self.tri_space1.is_boundary_dof()
        isDDof2 = self.imesh.boundary_face_flag()
        isDDof3 = self.int_space1.is_boundary_dof()

        bd_dof0 = ~((~isDDof1[:, None]) * (~isDDof3[None, :])).ravel()
        bd_dof1 = ~((~isDDof0[:, None]) * (~isDDof2[None, :])).ravel()

        return (bd_dof1, bd_dof0)

    def interpolation_points(self):
        ipoint0 = self.imesh.interpolation_points(p=1)
        ipoint1 = self.imesh.interpolation_points(p=2)
        ipoint2 = self.tmesh.interpolation_points(p=1)
        ipoint3 = self.tmesh.interpolation_points(p=2)
        
        p0 = bm.concat([bm.repeat(ipoint2, ipoint0.shape[0], axis=0), 
                          bm.tile(ipoint0.T, ipoint2.shape[0]).T], axis=1)
        p1 = bm.concat([bm.repeat(ipoint3, ipoint1.shape[0], axis=0), 
                          bm.tile(ipoint1.T, ipoint3.shape[0]).T], axis=1)
        
        return (p0, p1)

    def apply_bc(self, stokes_operator: StokesOperator, F):
        uh = self.x0
        gd = (self.velocity_dirichlet, self.pressure_dirichlet)
        threshold = (self.is_velocity_boundary, self.is_pressure_boundary)
        
        dofs = self.boundary_dof_index()
        points = self.interpolation_points()
        bd_dof = bm.concat([dofs[1], dofs[1], dofs[1], dofs[0]], axis=0)
        basic = [3*len(points[1]), 0]
        BdDof = []

        for i in range(2):
            index_dof = bm.arange(len(points[i]))[dofs[i]] + basic[i]
            # ipoints: (793， 3), 边界插值点坐标, 
            bd_point = points[i][dofs[i]] 
            # flag: (793,), 判断边界点是否属于某类边界
            flag = threshold[1-i](bd_point)
            # import ipdb;ipdb.set_trace()
            index_dof = index_dof[flag]
            val = gd[1-i](bd_point[flag])
            if i == 1:
                index_dof = bm.concat([index_dof, index_dof + len(points[1]), 
                                    index_dof + 2*len(points[1])], axis=0)
                val = val.reshape(-1, order='F')
            BdDof.append(index_dof)
            isBdDof = bm.zeros(self.n_A, dtype=bm.bool)
            isBdDof = bm.set_at(isBdDof, index_dof, True)
            uh = bm.set_at(uh, (..., isBdDof), val)
            
        # import ipdb;ipdb.set_trace()
        BdDof = bm.concat(BdDof)
        F = F - stokes_operator @ uh
        F = bm.set_at(F, BdDof, uh[BdDof])

        return BdDof, F

    def set_mesh(self, tmesh: TriangleMesh, imesh: IntervalMesh):
        self.tmesh = tmesh
        self.IMP = tmesh.uniform_refine(n=self.level-1, returnim=True)
        self.IMU = self.transfer(tmesh)
        self.imesh = imesh
        self.Ny = self.imesh.number_of_nodes()
        self.Nx = self.tmesh.number_of_nodes()
        self.NxNy = self.Nx * self.Ny
        
        tnode = tmesh.entity('node')
        inode = imesh.entity('node')
        
        self.node = bm.concat([bm.repeat(tnode, inode.shape[0], axis=0), 
                          bm.tile(inode.T, tnode.shape[0]).T], axis=1)

    def set_space_degree(self, p: int = 1) -> None:
        self.p = p

    def transfer(self, mesh: TriangleMesh):
        # input: the coaresest grid
        def P2red(NTc, c2i0, c2i1, Ndofc, Ndoff):
            # we just consider the middle points in fine edges. 
            elem2node2f = bm.zeros((NTc, 9), dtype=bm.int32)

            elem2node2f[:, 0] = c2i1[:NTc, 1]
            elem2node2f[:, 1] = c2i1[:NTc, 2]
            elem2node2f[:, 2] = c2i1[:NTc, 4]

            elem2node2f[:, 3] = c2i1[NTc:2*NTc, 1]
            elem2node2f[:, 4] = c2i1[NTc:2*NTc, 2]
            elem2node2f[:, 5] = c2i1[NTc:2*NTc, 4]

            elem2node2f[:, 6] = c2i1[2*NTc:3*NTc, 1]
            elem2node2f[:, 7] = c2i1[2*NTc:3*NTc, 2]
            elem2node2f[:, 8] = c2i1[2*NTc:3*NTc, 4]

            locRij = bm.array([
                [ 0,      3/8,   3/8,   0,     -1/8,   -1/8,    0,     -1/8,  - 1/8],
                [-1/8,    0,    -1/8,   3/8,    0,      3/8,   -1/8,    0,     -1/8],
                [-1/8,   -1/8,   0,    -1/8,   -1/8,    0,      3/8,    3/8,    0],
                [ 1/4,    0,     0,     3/4,    1/2,    0,      3/4,    0,      1/2],
                [ 1/2,    3/4,   0,     0,      1/4,    0,      0,      3/4,    1/2],
                [ 1/2,    0,     3/4,   0,      1/2,    3/4,    0,      0,      1/4]
            ])

            ii = bm.zeros((40*NTc,), dtype=bm.int32)
            jj = bm.zeros((40*NTc,), dtype=bm.int32)
            ss = bm.zeros((40*NTc,), dtype=bm.float64)
            index = 0

            for i in range(6):
                for j in range(9):
                    if locRij[i, j] != 0:
                        ii[index:index+NTc] = c2i0[:,i] + 1
                        jj[index:index+NTc] = elem2node2f[:,j] + 1
                        
                        # Every interior edges of the coarse grid will be computed twice.
                        # So the corresponding weight is halved.
                        if (j == 0) | (j == 4) | (j == 8):
                            ss[index:index+NTc] = locRij[i, j]
                        else:
                            ss[index:index+NTc] = locRij[i, j]/2
                        
                        index = index + NTc
            
            ii = ii[~(ii==0)] - 1
            jj = jj[~(jj==0)] - 1
            ss = ss[~(ss==0)] 
        
            # Modification for boundary middle points.
            idx = bm.concat([c2i1[:, 1], c2i1[:, 2], c2i1[:, 4]], axis=0)
            s = bm.bincount(idx, minlength=Ndoff)
            bdEdgeidxf = (s == 1)
            bdEdgeidxfinjj = bdEdgeidxf[jj]
            ss[bdEdgeidxfinjj] = 2*ss[bdEdgeidxfinjj]
        
            # The transfer operator for 1-6 nodes in the coarse grid is an identical matrix
            ii = bm.concat([bm.arange(Ndofc), ii], axis=0)
            jj = bm.concat([bm.arange(Ndofc), jj], axis=0)
        
            ss = bm.concat([bm.ones(Ndofc,), ss], axis=0)
            Pro = csr_matrix((ss, (jj, ii)), shape=(Ndoff, Ndofc))
            # space = LagrangeFESpace(mesh, p=2)
            # FreeNode = bm.nonzero(space.is_boundary_dof())[0]
            # if FreeNode is not None:
            #     FreeNonec = FreeNode[FreeNode<=Ndofc-1]
            #     Pro = Pro[FreeNode, FreeNonec]

            return Pro
        
        Pro_u = [None]*(self.level-1)
        for i in range(self.level-1):
            NTc = mesh.number_of_cells()
            c2i0 = mesh.cell_to_ipoint(p=2)
            Ndofc = mesh.number_of_global_ipoints(p=2)
            
            mesh.uniform_refine()
            c2i1 = mesh.cell_to_ipoint(p=2)
            Ndoff = mesh.number_of_global_ipoints(p=2)
            Pro_u[i] = P2red(NTc, c2i0, c2i1, Ndofc, Ndoff)

        return Pro_u
    
    def setup(self, A, B):
        """Compute restriction and interpolation operators.
        """
        level = self.level
        Ai = [None] * level
        Bi = [None] * level
        bigAi = [None] * level
        
        Ai[-1] = A
        Bi[-1] = B
        bigAi[-1] = [sp.bmat([[A, B.T],[B, None]])]

        Nu = bm.zeros((level,), dtype=bm.int32)
        Np = bm.zeros((level,), dtype=bm.int32)
        Nu[-1] = Ai[-1].shape[0] // 3
        Np[-1] = Bi[-1].shape[0]

        # Compute Pro and Res of u and p.
        Pro_u = self.IMU
        Pro_p = self.IMP

        for j in range(level - 1, 1, -1):
            # Ac = Res*Af*Pro
            Ai[j-1] = (sp.block_diag(Pro_u[j].T,Pro_u[j].T) @ self.A[-1] @ sp.block_diag(Pro_u[j-1],Pro_u[j-1]))
            Bi[j-1] = (Pro_p[j].T @ self.B[-1]*sp.block_diag(Pro_u[j-1],Pro_u[j-1]))
            Nu[j-1] = Ai[j-1].shape[0]
            Np[j-1] = Bi[j-1].shape[0]
            bigAi[j-1] = (sp.bmat([[self.A[-1], self.B[-1].T],[self.B[-1], None]]))

        Ndof = 3*Nu + Np
        auxMat = [None] * level

        for k in range(1, level): 
            Bt = Bi[k].T
            BBt = Bi[k] @ Bt
            BABt = Bi[k] @ Ai[k] @ Bt
            Su = bm.tril(Ai[k])
            Sp = bm.tril(BBt)
            Spt = bm.triu(BBt)
            DSp = sp.spdiags(BBt)
            auxMat[k] = {
                'Bt': Bt,
                'BBt': BBt,
                'BABt': BABt,
                'Su': Su,
                'Spt': Spt,
                'Sp': Sp,
                'DSp': DSp
            }
        self.Ai = Ai
        self.Bi = Bi
        self.Nu = Nu
        self.Np = Np
        self.Ndof = Ndof
        self.bigAi = bigAi
        self.auxMat = auxMat

    def vcycle(self, r, J):
        if J == 0:
            e = bm.zeros_like(r)
            e[:-1] = self.bigA[J][:-1, :-1] / r[:-1]
            e[3*self.Nu[J]:] -= bm.mean(e[3*self.Nu[J]])
            return e

        ru = r[:3*self.Nu[J]]
        rp = r[3*self.Nu[J]:]

        # pre-smoothing
        [eu, ep] = self.smoothing(bm.zeros((3*self.Nu[J],)),bm.zeros((self.Np[J],)),ru,rp,J)

        # form residual and restrict onto coarse grid
        rru = ru - self.A[J] @ eu - self.B[J].T @ ep
        rrp = rp - self.B[J] @ eu
        ruc = (self.Res_u[J] @ rru.reshape(self.Nu[J],3)).reshape(3*self.Nu[J], 1)
        rpc = self.Res_p[J] @ rrp
        
        # coarse grid correction
        rc = bm.concat([ruc, rpc], axis=0)
        ec = self.vcycle(rc, J-1)

        # correction on the fine grid
        tempeu = (self.Pro_u[J-1]@(ec[:3*self.Nu[J-1]].reshape(self.Nu[J-1],2))).reshape(3*self.Nu[J-1],)
        tempep = self.Pro_p[J-1]@ec[3*self.Nu[J-1]]
        eu = tempeu + eu
        ep = tempep + ep

        # post-smoothing
        [eu, ep] = self.smoothing(eu,ep,ru,rp,J)
        e = bm.concat([eu, ep], axis=0)
        return e       

    def wcycle(self, r, J=None): 
        if J is None:
            J = self.level
        if J == 0:
            e = self.A[J] / r
            return e
        
        ri = [None] * J
        ei = [None] * J

        # fine grid pre-smoothing
        e = self.B[J] / r
        for s in range(self.mu):
            e = e + self.B[J] / (r - self.A[J]@e)

        # restriction
        rc = self.R[J-1] @ (r - self.A[J-1]@e)

        # coarse grid correction twice
        ec = self.wcycle(rc, J-1)
        ec = ec + self.wcycle(rc - self.A[J-2]@ec, J-1)

        # prolongation
        e = e + self.P[J-2] @ ec

        # fine grid post-smoothing
        e = e + self.BBi[J-1] / (r - self.Ai[J-1]@e)
        for s in range(self.mu):
            e = e + self.BBi[J] / (r - self.Ai[J-1]@e)
    
    def smoothing(self, u, p, f, g, A, B):
        """Solve LUe = r.
        """
        auxMat = self.auxMat
        smootherOpt = self.options
        u, p = StokesLSCDGS(u,p,f,g,A,B,auxMat,smootherOpt)
        return u, p
    
    @variantmethod('direct')
    def solve(self, stokes_operator: StokesOperator, F, solver='mumps'):
    # def solve(self, A, F, solver='mumps'):
        """
        Solve the linear system using direct method.
        """
        self.solve_str = 'direct'
        # import pyamg
        # solver = pyamg.ruge_stuben_solver(stokes_operator)
        # x = solver.solve(F, tol=1e-10)
        from fealpy.solver import bicgstab, minres, gmres, cg, bicg, lgmres
        x, info = bicgstab(stokes_operator, F)
        # x, info = gmres(stokes_operator, F)
        # x, info = minres(stokes_operator, F)
        # x, info = cg(stokes_operator, F, returninfo=True)
        # x, info = bicg(stokes_operator, F)
        # x, info = lgmres(stokes_operator, F)
        print(info)
        return x
        # return spsolve(A, F, solver = solver)

    @solve.register('mg')
    def solve(self, A, B, f, g, u, p):
        # initial set up
        self.setup(A, B)

        bigF = bm.concat([f,g-bm.mean(g)], axis=0)
        bigu = bm.concat([u, p], axis=0)
        bigr = bigF - self.bigAi[-1] @ bigu

        k = 0
        nb = bm.linalg.norm(bigF)
        err = bm.zeros((self.maxIt, 1), dtype=bm.float64)
        err[0] = bm.linalg.norm(bigr) / nb

        while (bm.max(err[k]) > self.tol) & (k <= self.maxIt):
            k = k + 1
            if self.solver == 'VCYCLE':
                bigerru = self.vcycle(bigr)
            elif self.solver == 'WCYCLE':
                bigerru = self.wcycle(bigr)
            bigu = bigu + bigerru
            bigr = bigr - self.bigAi[-1] @ bigerru

            # compute the relative error
            err[k] = bm.linalg.norm(bigr) / nb

            print(
                f'MG Vcycle iter: {k:2d},   '
                f'err = {bm.max(err[k, :]):8.4e}'
            )
        err = err[:k]
        itStep = k
        u = bigu[:3*self.Nu[-1]]
        p = bigu[3*self.Nu[-1]:]

        # Output
        print(f"iter: {itStep:2.0f},  "
            f"err = {max(err[-1]):8.4e},  "
            f"coarse grid: {self.A[-1].shape[0]:2.0f},  ")

        if k > self.maxIt:
            print("NOTE: the iterative method does not converge!")

        return u, p

    @solve.register('asmg')
    def solve(self, A, F):
        raise NotImplementedError("AMG solver not yet implemented.")

    def run(self):
        tmr = timer()
        next(tmr)
        print(f'1. 初始化完成，开始组装线性系统')
        stokes_operator, F = self.linear_system()
        # A, F = self.linear_system()
        print(f'2. 初步组装线性系统完成')
        tmr.send(f'初步组装线性系统时间')
        BdDof, F0 = self.apply_bc(stokes_operator, F)
        stokes_operator.is_boundary_dof = BdDof
        # BC = DirichletBC(
        #     (self.uspace, self.pspace),
        #     gd=(self.velocity_dirichlet, self.pressure_dirichlet),
        #     threshold=(self.is_velocity_boundary, self.is_pressure_boundary),
        #     method='interp'
        # )
        # A, F1 = BC.apply(A, F)
        # import ipdb;ipdb.set_trace()
        print(f'3. 作用边界条件完成')
        tmr.send(f'作用边界条件时间')
        import time
        start = time.time()
        # x = self.solve(A, F0)
        x = self.solve(stokes_operator, F0)
        print(f'4. 求解完成')
        tmr.send(f'求解器时间')
        next(tmr)
        print(time.time() - start)
        ugdof = self.n_u0
        uh = x[:ugdof]
        # import ipdb;ipdb.set_trace()
        ph = x[ugdof:]
        # err = self.postprocess()
        # self.logger.info(f"L2 point Error: {err}.")
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
        # import ipdb;ipdb.set_trace()
        idx = bm.arange(gdof).reshape(tgdof, -1)[:tNN, :iNN].ravel()

        # from fealpy.mesh import TensorPrismMesh
        # self.mesh = TensorPrismMesh(self.tmesh, self.imesh)
        # self.mesh.nodedata['ph'] = ph
        # self.mesh.nodedata['uh'] = uh.reshape(3,-1).T[idx,:]
        
        # self.mesh.to_vtk('dld_prism_chip.vtu')