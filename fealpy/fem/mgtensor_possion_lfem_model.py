from typing import Optional, Union
from fealpy.backend import bm

from fealpy.mesh import TriangleMesh, IntervalMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm, LinearForm, DirichletBC
from fealpy.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator, ScalarSourceIntegrator
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.model.mgtensor_possion import MGTensorPossionPDEDataT

from fealpy.sparse import spdiags, coo_matrix, csr_matrix
from fealpy.solver import cg, spsolve, transferP1red

from fealpy.utils import timer
from fealpy.decorator import variantmethod

import scipy.sparse as sp
import scipy.sparse.linalg as lg

from petsc4py import PETSc

import time
import gc


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
        Y = self.A @ X @ self.B
        Y = Y.ravel()
        return Y


class PoissonOperator(LinearOperator):
    def __init__(self, Ax, Mx, Az, Mz):
        self.Ax = Ax
        self.Mx = Mx
        self.Az = Az
        self.Mz = Mz
        self.set_up()

    def set_up(self):
        self.n_Ax = self.Ax.shape[0]
        self.n_Mz = self.Mz.shape[0]
        self.n_A = self.n_Ax * self.n_Mz
        self.shape = self.n_A, self.n_A

    def assembly(self):
        return sp.kron(self.Ax, self.Mz) + sp.kron(self.Mx, self.Az)

    def __matmul__(self, x):
        v = bm.copy(x)
        X = bm.reshape(v, (self.n_Ax, self.n_Mz))
        Y = self.Ax @ X @ self.Mz + self.Mx @ X @ self.Az
        Y = Y.ravel()
        return Y


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
        # 直接计算每个块，避免构建大矩阵 U 和 full matrix multiplication
        Y1 = self.Ax @ (bm.reshape(v[:n], (self.n0, self.m1)) @ self.Mz) + self.Mx @ (bm.reshape(v[:n], (self.n0, self.m1)) @ self.Az)
        Y2 = self.Ax @ (bm.reshape(v[n:2*n], (self.n0, self.m1)) @ self.Mz) + self.Mx @ (bm.reshape(v[n:2*n], (self.n0, self.m1)) @ self.Az)
        Y3 = self.Ax @ (bm.reshape(v[2*n:3*n], (self.n0, self.m1)) @ self.Mz) + self.Mx @ (bm.reshape(v[2*n:3*n], (self.n0, self.m1)) @ self.Az)
        return [Y1.ravel(), Y2.ravel(), Y3.ravel()]


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


class MGTensorPossionLFEMModel(ComputationalModel):
    """"Multigrid solver for Poisson equations defined on 
            tensor-product grids using the Linear Finite Element Method (LFEM).
    """
    def __init__(self, options=None):
        super().__init__(pbar_log=True, log_level="INFO")
        self.pdm = PDEModelManager("mgtensor_possion")

        if options is None:
            options = {} 
            
        self.level = options.get('level')

        self.options = options
        self.x0 = options.get('x0', None)
        self.tol = options.get('tol', 1e-8)  
        self.maxIt = options.get('solvermaxit', 200)  
        self.N0 = options.get('N0', 500)
        self.mu = options.get('smoothingstep', 1)
        self.solver = options.get('solver', 'mg')

        self.cycle_type = options.get('cycle_type', 'VCYCLE')
        self.preconditioner = options.get('preconditioner', 'none')
        self.coarsegridsolver = options.get('coarsegridsolver', 'direct')
        self.smoother = options.get('smoother', 'LINE')
    
        self.coarse_time = 0

    def set_pde(self, pde: Union[MGTensorPossionPDEDataT, str, int]):
        """
        """
        if isinstance(pde, int):
            self.pde = self.pdm.get_example(pde)
        else:
            self.pde = pde

    def set_mesh(self, tmesh: TriangleMesh, imesh: IntervalMesh):
        self.mesh0 = TriangleMesh(tmesh.entity('node'), tmesh.entity('cell'))
        self.mesh1 = TriangleMesh(tmesh.entity('node'), tmesh.entity('cell'))
        tmesh.uniform_refine(self.level-1)
        self.tmesh = tmesh
        self.imesh = imesh
        self.Ny = self.imesh.number_of_nodes()
        self.Nx = self.tmesh.number_of_nodes()
        self.NxNy = self.Nx * self.Ny
        self.x0 = bm.zeros((self.NxNy,), dtype=bm.float64)
        
        tnode = tmesh.entity('node')
        inode = imesh.entity('node')
        
        self.node = bm.concat([bm.repeat(tnode, inode.shape[0], axis=0), 
                          bm.tile(inode.T, tnode.shape[0]).T], axis=1)

    def set_space_degree(self, p: int = 1) -> None:
        self.p = p

    def linear_system(self):
        """
        """
        from fealpy.mesh import TensorPrismMesh
        self.mesh = TensorPrismMesh(self.tmesh, self.imesh)
        self.space = LagrangeFESpace(self.mesh, p=1)

        p = self.p
        self.space0= LagrangeFESpace(self.tmesh, p=p)
        self.space1= LagrangeFESpace(self.imesh, p=p)

        form00 = BilinearForm(self.space0)
        form00.add_integrator(ScalarDiffusionIntegrator())
        Ax = form00.assembly().to_scipy()

        form01 = BilinearForm(self.space0)
        form01.add_integrator(ScalarMassIntegrator())
        Mx = form01.assembly().to_scipy()

        form10 = BilinearForm(self.space1)
        form10.add_integrator(ScalarDiffusionIntegrator())
        Az = form10.assembly().to_scipy()

        form11 = BilinearForm(self.space1)
        form11.add_integrator(ScalarMassIntegrator())
        Mz = form11.assembly().to_scipy()

        gdof = Ax.shape[0]*Mz.shape[0]
        print(f'自由度个数：{gdof}')
        op = PoissonOperator(Ax, Mx, Az, Mz)

        A = (sp.kron(Ax, Mz) + sp.kron(Mx, Az)).tocoo()
        from fealpy.sparse import COOTensor
        A = COOTensor(
            indices=bm.stack([A.row, A.col], axis=0),
            values=A.data,
            spshape=A.shape
        )

        # A = None
        self.x0 = bm.zeros((gdof,), dtype=bm.float64)
        F = bm.zeros((gdof,), dtype=bm.float64)
        self.total_dof = gdof
        return op, A, F

    def apply_bc(self, op: PoissonOperator, F):
        isDDof0 = self.space0.is_boundary_dof()
        isDDof1 = self.imesh.boundary_face_flag()
        index_dof = bm.arange(self.NxNy)[~((~isDDof0[:, None]) * (~isDDof1[None, :])).ravel()]
        gd = self.pde.dirichlet
        threshold = self.pde.is_dirichlet_boundary
        uh = self.x0
        ipoint = self.node[index_dof]
        flag = threshold(ipoint)
        
        index_dof = index_dof[flag]
        isBdDof = bm.zeros(self.NxNy, dtype=bm.bool)
        isBdDof = bm.set_at(isBdDof, index_dof, True)

        gd = gd(self.node[isBdDof])
        uh = bm.set_at(uh, (..., isBdDof), gd)
    
        F = F - op @ uh # 5000w ~ 400MB
        F = bm.set_at(F, isBdDof, uh[isBdDof])

        # Fixdof
        inflag0 = ~isDDof0
        inflag1 = ~isDDof1

        op.Ax = op.Ax[inflag0][:,inflag0]
        op.Mx = op.Mx[inflag0][:,inflag0]
        op.Az = op.Az[inflag1][:,inflag1]
        op.Mz = op.Mz[inflag1][:,inflag1]
        
        op.set_up()

        return op, F, isBdDof

    def setup(self, op: PoissonOperator):
        """Compute restriction and interpolation operators.
        """
        Ax = op.Ax
        Mx = op.Mx
        Az = op.Az
        Mz = op.Mz

        level = self.level
        Axi = [None] * level
        Mxi = [None] * level
        Axi[-1] = Ax
        Mxi[-1] = Mx

        # Compute P and R.
        P = transferP1red(self.mesh0, self.level, self.pde.is_dirichlet_boundary, tensor_mesh=True)
        Iz = spdiags(bm.ones((self.Ny,)), 0, self.Ny, self.Ny)
        
        for j in range(self.level - 1, 0, -1):
            Axi[j-1] = P[j-1].T @ Axi[j] @ P[j-1]
            Mxi[j-1] = P[j-1].T @ Mxi[j] @ P[j-1]
            
        self.P = [None] * (level-1)
        self.R = [None] * (level-1)

        self.Ai = [None] * level
        self.Bi = [None] * level
        self.Li = [None] * level
        self.Ri = [None] * level
        self.bigAi = [None] * level

        Iz = spdiags(bm.ones((op.n_Mz,)), 0, op.n_Mz, op.n_Mz).to_scipy()
        
        for j in range(self.level):
            self.Ai[j] = PoissonOperator(Axi[j], Mxi[j], Mz, Az)
            self.Bi[j] = sp.kron(sp.diags(Axi[j].diagonal()), Mz) + \
                        sp.kron(sp.diags(Mxi[j].diagonal()), Az)
            # self.Li[j] = lg.splu(self.Bi[j], )
            if j < self.level - 1:
                self.P[j] = KronOperator(P[j], Iz)
                self.R[j] = KronOperator(P[j].T, Iz)
        # import ipdb;ipdb.set_trace()
        self.Ai[0] = self.Ai[0].assembly()
    
    def coarse_solve(self, r):
        
        pass

    def vcycle(self, r, J=None):
        """solve equations Ae = r in each level  
        """   
        if J is None:
            J = self.level
        
        ri = [None] * J
        ei = [None] * J
        ri[-1] = r

        for i in range(J-1,0,-1):
            ei[i] = self.linesmoother(ri[i], i)

            for _ in range(self.mu):
                ei[i] += self.linesmoother(ri[i] - self.Ai[i] @ ei[i], i)
            
            ri[i-1] = self.R[i-1] @ (ri[i] - self.Ai[i] @ ei[i])

        if self.coarsegridsolver == 'direct':
            start = time.time()
            ei[0] = self.solve(self.Ai[0], ri[0])
            self.coarse_time += time.time() - start
        else:
            pass
        
        for i in range(J-1):          
            ei[i+1] += self.P[i] @ ei[i]
            ei[i+1] += self.linesmoother(ri[i+1] - self.Ai[i+1] @ ei[i+1], i+1)

            for _ in range(self.mu):
                ei[i+1] += self.linesmoother(ri[i+1] - self.Ai[i+1] @ ei[i+1], i+1)
        
        return ei[-1]

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

    def linesmoother(self, r, J):
        """Solve LUe = r.
        """
        e = cg(self.Bi[J], r, maxit=100, atol=1e-6, rtol=1e-6)
        # e = spsolve(self.B[J], r)
        e = 0.75 * e

        return e
    
    @variantmethod('cg')
    def solve(self, A, F):
        import ipdb;ipdb.set_trace()
        x, info = cg(A, F, maxit=1000, atol=1e-9, rtol=1e-9, returninfo=True)
        print(info)
        return x
    
    @solve.register('mg')
    def solve(self, op: PoissonOperator, F):
        # initial set up
        self.setup(op)
        x = bm.zeros_like(len(F))

        k = 0
        r = F
        nb = bm.linalg.norm(F)
        err = bm.zeros((self.maxIt, 2), dtype=bm.float64)

        if nb > bm.finfo(float).eps:
            err[0, :] = bm.linalg.norm(r) / nb
        else:
            err[0, :] = bm.linalg.norm(r)

        if self.cycle_type == 'VCYCLE':
            print('Multigrid Vcycle Iteration \n')
            while (bm.max(err[k, :]) > self.tol) & (k <= self.maxIt):
                k = k + 1
                Br = self.vcycle(r)
                x = x + Br
                r = r - op @ Br
                err[k, 0] = bm.sqrt(bm.abs(Br.T @ r / (x.T @ F)))
                err[k, 1] = bm.linalg.norm(r) / nb

                print(
                    f'MG Vcycle iter: {k:2d},   '
                    f'err = {bm.max(err[k, :]):8.4e}'
                )
            err = err[:k, :]
            itStep = k

        elif self.cycle_type == 'WCYCLE':
            pass
        
        # Output
        print(f"dof: {self.NxNy:2.0f},  "
            f"level: {self.level:2.0f},  "
            f"smoothing: {self.mu:2.0f},  "
            f"iter: {itStep:2.0f},  "
            f"err = {max(err[-1]):8.4e},  "
            f"coarse grid: {self.Ai[0].shape[0]:2.0f},  ")

        if k > self.maxIt:
            print("NOTE: the iterative method does not converge!")

        return x

    def run(self):
        import time
        start = time.time()
        op0, A, F = self.linear_system()
        self.logger.info(f'Step 1. 完成初步线性系统组装\n')
        op, F1, BdDof = self.apply_bc(op0, bm.copy(F))
        
        del op0
        gc.collect()
        self.logger.info(f'Step 2. 完成边界自由度处理\n')
        
        self.solver = 'mg'
        # self.solver = 'direct'
        if self.solver == 'direct':
            BC = DirichletBC(self.space, gd=self.pde.dirichlet,
                threshold=self.pde.is_dirichlet_boundary,
                method='interp'
            )
            A, F2 = BC.apply(A, F)
            print(f'开始求解')
            tmr = timer()
            next(tmr)
            # x = spsolve(A.to_scipy(), F2)
            x = self.solve['direct'](A.to_scipy(), F2)
            tmr.send(f'求解器时间')
            next(tmr)
            
        elif self.solver == 'mg':            
            bd_flag = bm.zeros((len(F),), dtype=bm.bool)
            bm.set_at(bd_flag, BdDof, True)
            self.logger.info(f'Step 3. 开始多重网格setup阶段\n')
            x_in = self.solve['mg'](op, F1[~bd_flag])
            x = bm.set_at(F1, ~bd_flag, x_in)
        
        print(x.max())
        print(self.post_process(x))
        return x
    
    def post_process(self, x):
        err = bm.sqrt(bm.mean((self.pde.solution(self.node) - x)**2))
        return err
    

