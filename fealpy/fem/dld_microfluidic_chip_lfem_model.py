from typing import Any, Optional, Union

from fealpy.backend import bm
from fealpy.typing import TensorLike
from fealpy.decorator import variantmethod, cartesian
from fealpy.model import ComputationalModel

from fealpy.mesh import TriangleMesh, LagrangeTriangleMesh
from fealpy.mesher import DLDMicrofluidicChipMesher
from fealpy.geometry.implicit_curve import CircleCurve
from fealpy.functionspace import functionspace, ParametricLagrangeFESpace, TensorFunctionSpace, LagrangeFESpace
from fealpy.fem import LinearForm, BilinearForm, BlockForm, LinearBlockForm
from fealpy.fem import ScalarDiffusionIntegrator as DiffusionIntegrator
from fealpy.fem import DirichletBC, StokesDirichletBC
from fealpy.fem import PressWorkIntegrator, SourceIntegrator
from fealpy.solver import StokesLSCDGS, cg, spsolve, transferP1red, transferP2red
from fealpy.sparse import spdiags, coo_matrix, csr_matrix

import scipy.sparse as sp
import scipy.sparse.linalg as lg
import time

class DLDMicrofluidicChipLFEMModel(ComputationalModel):
    """
    A Lagrange finite element computational model class for Deterministic Lateral 
    Displacement (DLD) microfluidic chip simulation.

    Parameters:
        options (dict, optional): A dictionary containing computational options 
            for the model. Expected keys include:
            - pbar_log: Whether to enable progress bar logging
            - log_level: Logging level for the model

    Attributes:
        pde (StokesPDEDataT): The PDE data object containing problem definition
        mesh: The computational mesh
        equation (StationaryStokes): The Stokes equation object
        fem: The finite element method implementation
        uspace: Velocity function space
        pspace: Pressure function space
        p (int): Polynomial degree for function spaces

    Methods:
        set_inlet_condition: Set the PDE data for the model
        set_init_mesh: Set the initial mesh
        setup: Initialize PDE equation and FEM method
        set_space_degree: Set polynomial degree for function spaces
        linear_system: Assemble the linear system
        solve: Solve the linear system
    """
    def __init__(self, options: dict = None):
        self.options = options
        super().__init__(
            pbar_log=options['pbar_log'],
            log_level=options['log_level']
        )
        if options is None:
            options = {} 
            
        self.level = options.get('level', 4)

        self.options = options
        self.x0 = options.get('x0', None)
        self.tol = options.get('tol', 1e-8)  
        self.maxIt = options.get('solvermaxit', 200)  
        self.N0 = options.get('N0', 500)
        self.mu = options.get('smoothingstep', 1)

        self.cycle_type = options.get('cycle_type', 'VCYCLE')
        self.smoothing_times = options.get('smoothing_times', 1)
        self.preconditioner = options.get('preconditioner', 'none')
        self.coarsegridsolver = options.get('coarsegridsolver', 'direct')
        self.solver = options.get('solver', 'mg')

        self.thickness = options.get('thickness', 0.1)

        self.coarse_time = 0
        self.smoothing_time = 0
        self.coarse_count = 0
        self.smoothing_count = 0

        self.mul_time = 0

    def set_init_mesher(self, mesher: DLDMicrofluidicChipMesher):
        """
        Set the initial mesh for the simulation.
        
        Parameters:
            mesh: The computational mesh object
        """
        mesh = mesher.mesh
        self.mesh0 = TriangleMesh(mesh.entity('node'), mesh.entity('cell'))
        self.mesh1 = TriangleMesh(mesh.entity('node'), mesh.entity('cell'))
        mesh.uniform_refine(self.level-1)
        self.mesh = mesh
        self.radius = mesher.radius
        self.centers = mesher.centers
        self.boundary = mesher.boundary
        self.inlet_boundary = mesher.inlet_boundary
        self.outlet_boundary = mesher.outlet_boundary
        self.wall_boundary = mesher.wall_boundary
        self.project_edges = mesher.project_edges

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
            result = bm.zeros(p.shape, dtype=bm.float64)
            result[..., 0] = y * (1-y)
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
            return self.is_boundary(p, bd)
        
        @cartesian
        def is_outlet_boundary( p: TensorLike) -> TensorLike:
            """Check if point where pressure is defined is on boundary."""
            bd = self.outlet_boundary
            return self.is_boundary(p, bd)

        @cartesian
        def is_wall_boundary(p: TensorLike) -> TensorLike:
            """Check if point where velocity is defined is on boundary."""
            bd = self.wall_boundary
            return self.is_boundary(p, bd)
        
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
        self.is_obstacle_boundary = is_obstacle_boundary

    def is_boundary(self, p: TensorLike, bd: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        atol = 1e-12
        v0 = p[:, None, :] - bd[None, 0::2, :] # (NN, NI, 2)
        v1 = p[:, None, :] - bd[None, 1::2, :] # (NN, NI, 2)

        cross = v0[..., 0]*v1[..., 1] - v0[..., 1]*v1[..., 0] # (NN, NI)
        dot = bm.einsum('ijk,ijk->ij', v0, v1) # (NN, NI)
        cond = (bm.abs(cross) < atol) & (dot < atol)
        return bm.any(cond, axis=1)
    
    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        inlet = self.is_inlet_boundary(p)
        wall = self.is_wall_boundary(p)
        obstacle = self.is_obstacle_boundary(p)
        return inlet | wall | obstacle
    
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
        GD = self.mesh.geo_dimension()
        self.uspace = functionspace(self.mesh, ('Lagrange', self.p), shape=(GD, -1))
        self.pspace = functionspace(self.mesh, ('Lagrange', self.p-1))
        print(self.uspace.number_of_global_dofs() + self.pspace.number_of_global_dofs())
        
        A00 = BilinearForm(self.uspace)
        self.BD = DiffusionIntegrator()
        self.BD.coef = 1.0
        A00.add_integrator(self.BD)
        A01 = BilinearForm((self.pspace, self.uspace))
        self.BP = PressWorkIntegrator()
        self.BP.coef = -1.0
        A01.add_integrator(self.BP)
        
        L0 = LinearForm(self.uspace)
        L1 = LinearForm(self.pspace)
        
        return A00, A01, L0, L1

    @linear_system.register('isopara')
    def linear_system(self):
        """
        Assemble the linear system for the Stokes equations.
        """
        edge = self.mesh.entity('edge')
        node = self.mesh.interpolation_points(self.p)
        cell = self.mesh.cell_to_ipoint(self.p)
        NN = self.mesh.number_of_nodes()
        for i in range(len(self.centers)):
            edge_flag = self.find_edge_indices(edge, self.project_edges[i])
            curve = CircleCurve(center=self.centers[i], radius=self.radius)
            isCircleNode = edge_flag + NN
            bdNode, _ = curve.project(node[isCircleNode])
            node = bm.set_at(node, isCircleNode, bdNode)

        mesh = LagrangeTriangleMesh(node, cell, p=self.p)
        space = ParametricLagrangeFESpace(mesh, p=self.p)
        GD = self.mesh.geo_dimension()

        self.uspace = TensorFunctionSpace(space, shape=(GD, -1))
        self.pspace = LagrangeFESpace(self.mesh, self.p-1)
        A00 = BilinearForm(self.uspace)
        A00.add_integrator(DiffusionIntegrator(coef=1.0, method='isopara'))
        A01 = BilinearForm((self.pspace, self.uspace))
        A01.add_integrator(PressWorkIntegrator(coef=-1.0, method='isopara'))
        A = BlockForm([[A00, A01], [A01.T, None]])
        L0 = LinearForm(self.uspace)
        L1 = LinearForm(self.pspace)
        L = LinearBlockForm([L0, L1])

        return A, L

    def setup(self, A, B):
        """Compute restriction and interpolation operators.
        """
        A = A.to_scipy()
        B = B.to_scipy()
        level = self.level
        Ai = [None] * level
        Bi = [None] * level
        bigAi = [None] * level
        
        Ai[-1] = A
        Bi[-1] = B
        
        bigAi[-1] = sp.bmat([[A, B.T],[B,None]]).tocsr()

        Nu = bm.zeros((level,), dtype=bm.int32)
        Np = bm.zeros((level,), dtype=bm.int32)
        Nu[-1] = Ai[-1].shape[0] // 2
        Np[-1] = Bi[-1].shape[0]

        # Compute Pro and Res of u and p.
        Pro_p = transferP1red(self.mesh0, self.level)
        Pro_u = transferP2red(self.mesh1, self.level)
        Res_u = [None] * level
        Res_p = [None] * level

        for i in range(self.level - 1):
            Pro_u[i] = sp.block_diag([Pro_u[i].to_scipy(),Pro_u[i].to_scipy()])
            Pro_p[i] = Pro_p[i].to_scipy()
            Res_u[i] = Pro_u[i].T
            Res_p[i] = Pro_p[i].T
        
        for j in range(level - 1, 0, -1):
            # Ac = Res*Af*Pro
            Ai[j-1] = Res_u[j-1] @ Ai[j] @ Pro_u[j-1]
            Bi[j-1] = Res_p[j-1] @ Bi[j] @ Pro_u[j-1]
            Nu[j-1] = Ai[j-1].shape[0] // 2
            Np[j-1] = Bi[j-1].shape[0]
            bigAi[j-1] = (sp.bmat([[Ai[j-1], Bi[j-1].T],[Bi[j-1], None]]).tocsr())

        Ndof = 2 * Nu + Np
        auxMat = [None] * level
        
        self.Pro_u = Pro_u
        self.Pro_p = Pro_p
        self.Res_u = Res_u
        self.Res_p = Res_p

        for k in range(1, level): 
            Bt = Bi[k].T
            BBt = Bi[k] @ Bt
            BABt = Bi[k] @ Ai[k] @ Bt
            Su = sp.tril(Ai[k]).tocsr()
            Sp = sp.tril(BBt).tocsr()
            Spt = sp.triu(BBt).tocsr()
            DSp = BBt.diagonal()
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

    def vcycle(self, r, J=None):
        if J is None:
            J = self.level - 1
        if J == 0:
            import pyamg
            start = time.time()
            # mg = pyamg.ruge_stuben_solver(self.bigAi[J])
            # e = mg.solve(r)
            e = spsolve(self.bigAi[J], r)
            self.coarse_count += 1
            self.coarse_time += time.time() - start
            return e
        
        Pro_u = self.Pro_u[J-1]
        Pro_p = self.Pro_p[J-1]
        Res_u = self.Res_u[J-1]
        Res_p = self.Res_p[J-1]
        
        ru = r[:2*self.Nu[J]]
        rp = r[2*self.Nu[J]:]
        
        # pre-smoothing
        eu, ep = self.smoothing(bm.zeros((2*self.Nu[J],)),bm.zeros((self.Np[J],)),ru,rp,J)
        if self.smoothing_times == 2:
            eu, ep = self.smoothing(eu,ep,ru,rp,J)

        # form residual and restrict onto coarse grid
        rru = ru - self.Ai[J] @ eu - self.Bi[J].T @ ep
        rrp = rp - self.Bi[J] @ eu

        ruc = Res_u @ rru
        rpc = Res_p @ rrp
        
        # coarse grid correction
        rc = bm.concat([ruc, rpc], axis=0)
        ec = self.vcycle(rc, J-1)

        # correction on the fine grid
        tempeu = Pro_u @ ec[:2*self.Nu[J-1]]
        tempep = Pro_p @ ec[2*self.Nu[J-1]:]
        eu = tempeu + eu
        ep = tempep + ep

        # post-smoothing
        eu, ep = self.smoothing(eu,ep,ru,rp,J)
        if self.smoothing_times == 2:
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
        u, p, self.mul_time = smoother.run(u,p,f,g,A,B,self.mul_time)
        t = time.time() - start
        print(t)
        self.smoothing_time += t
        self.smoothing_count += 1
        return u, p

    @variantmethod('direct')
    def solve(self, A, F, solver='mumps'):
        """
        Solve the linear system using direct method.
        """
        self.solve_str = 'direct'
        return spsolve(A, F, solver = 'mumps')

    @solve.register('mg')
    def solve(self, A, B, f, g):
        # initial set up
        u = bm.zeros_like(f)
        p = bm.zeros_like(g)
        self.setup(A, B.T)
        
        bigF = bm.concat([f,g-bm.mean(g)], axis=0)
        bigu = bm.concat([u, p], axis=0)
        bigr = bigF - self.bigAi[-1] @ bigu

        k = 0
        nb = bm.linalg.norm(bigF)
        err = bm.zeros((self.maxIt, 1), dtype=bm.float64)
        err[0] = bm.linalg.norm(bigr) / nb
        print(f'2. set_up完成, 开始执行多重网格方法\n')
        start = time.time()
        while (bm.max(err[k]) > self.tol) & (k < self.maxIt - 1):
            k = k + 1
            if self.cycle_type == 'VCYCLE':
                bigerru = self.vcycle(bigr)
            elif self.cycle_type == 'WCYCLE':
                bigerru = self.wcycle(bigr)
            bigu = bigu + bigerru
            bigr = bigr - self.bigAi[-1] @ bigerru

            # compute the relative error
            err[k] = bm.linalg.norm(bigr) / nb

            print(
                f'MG {self.cycle_type} iter: {k:2d},   '
                f'err = {err[k][0]:8.4e},   '
                f'Ndof = {self.Ndof[-1]}\n'
            )
        err = err[:k+1]
        itStep = k
        u = bigu[:2*self.Nu[-1]]
        p = bigu[2*self.Nu[-1]:]
        cost = time.time() - start
        # Output
        print(f"iter: {itStep:2.0f},  "
            f"err = {err[-1][0]:8.4e}\n"
            f"level = {self.level}\n"
            f"coarse grid: {self.Bi[0].shape[0]:2.0f}\n"
            f"coarse dofs: {self.Ndof[0]:2.0f}\n"
            f"total time in coarsest grid: {self.coarse_time}\n"
            f"total time: {cost}\n"
            f"points: {self.coarse_time / cost},  "
            f"coarse count: {self.coarse_count}")

        if k > self.maxIt:
            print("NOTE: the iterative method does not converge!")

        return u, p

    @variantmethod('one_step')
    def run(self):
        A00, A01, L0, L1 = self.linear_system()
        pdof = self.pspace.number_of_global_dofs()
        if self.solver == 'direct':
            bigA0 = BlockForm([[A00, A01], [A01.T, None]]).assembly()
            bigF0 = LinearBlockForm([L0, L1]).assembly()

            BC = DirichletBC(
                (self.uspace, self.pspace), 
                gd=self.velocity_dirichlet,
                threshold=self.is_velocity_boundary,
                method='interp'
            )

            bigA0, bigF0 = BC.apply(bigA0, bigF0)
            bigF0[-pdof:] = bigF0[-pdof:] - bm.mean(bigF0[-pdof:])
            bigA0 = bigA0.to_scipy()
            import time
            from fealpy.solver import gmres, lgmres
            start = time.time()
            x = spsolve(bigA0, bigF0, 'mumps')
            # x, _ = gmres(bigA0, bigF0)
            print(f'direct time: {time.time() - start}')
            uh = x[:-pdof]
            ph = x[-pdof:]

        elif self.solver == 'mg':
            A = A00.assembly()
            B = A01.assembly()
            f = L0.assembly()
            g = L1.assembly()
            StokesBC = StokesDirichletBC(
                (self.uspace, self.pspace),
                gd=self.velocity_dirichlet,
                threshold=self.is_velocity_boundary,
                method='interp'
            )

            A, B, f, g = StokesBC.apply(A, B, f, g)
            print(f'1. 组装线性系统完成')
            uh, ph = self.solve['mg'](A, B, f, g)        
        
        print(f'smoothing_time:{self.smoothing_time}')
        print(f'smoothing_count:{self.smoothing_count}')
        print(f'spsolve_triangular_time:{self.mul_time}')
        self.post_process(uh ,ph)
    
    def post_process(self, uh, ph):
        self.mesh.nodedata['ph'] = ph
        self.mesh.nodedata['uh'] = uh.reshape(2,-1).T
        self.mesh.to_vtk('dld_chip.vtu')
    
    def find_edge_indices(self, edge, inedge):
        inedge_sorted = bm.sort(inedge, axis=1)   # (N,2)
        edge_sorted = bm.sort(edge, axis=1)     # (NE,2)
        edge_dict = {tuple(e): i for i, e in enumerate(edge_sorted)}
        indices = bm.array([edge_dict.get(tuple(e), -1) for e in inedge_sorted])
        return indices
    