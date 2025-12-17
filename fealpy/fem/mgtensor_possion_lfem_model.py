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

        self.setup_time = 0
        self.coarse_time = 0
        self.smoothing_time = 0
        self.MUL_time = 0

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
            self.Ai[j] = PoissonOperator(Axi[j], Mxi[j], Az, Mz)
            self.Bi[j] = sp.kron(sp.diags(Axi[j].diagonal()), Mz) + \
                        sp.kron(sp.diags(Mxi[j].diagonal()), Az)
            # self.Li[j] = lg.splu(self.Bi[j], )
            if j < self.level - 1:
                self.P[j] = KronOperator(P[j], Iz)
                self.R[j] = KronOperator(P[j].T, Iz)

        self.coarse_dof = self.Ai[0].shape[0]
        A = self.Ai[0].assembly().tocsr().astype(bm.float64)
        A = csr_to_petsc_mat(A)

        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setType('cg') 
        ksp.getPC().setType('gamg')

        ksp.setFromOptions()
        ksp.setUp()

        self.A0 = ksp
    
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
                start = time.time()
                ra = ri[i] - self.Ai[i] @ ei[i]
                self.MUL_time += time.time() - start

                ei[i] += self.linesmoother(ra, i)
            
            start = time.time()
            ri[i-1] = self.R[i-1] @ (ri[i] - self.Ai[i] @ ei[i])
            self.MUL_time += time.time() - start

        if self.coarsegridsolver == 'direct':
            start = time.time()
            ei[0] = self.solve['pets'](ri[0])
            self.coarse_time += time.time() - start
        else:
            pass
        
        for i in range(J-1):    
            start = time.time()      
            ei[i+1] += self.P[i] @ ei[i]
            rb = ri[i+1] - self.Ai[i+1] @ ei[i+1]
            self.coarse_time += time.time() - start

            ei[i+1] += self.linesmoother(rb, i+1)

            for _ in range(self.mu):
                start = time.time()
                rc = ri[i+1] - self.Ai[i+1] @ ei[i+1]
                self.coarse_time += time.time() - start

                ei[i+1] += self.linesmoother(rc, i+1)
        
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
        start = time.time()
        e = cg(self.Bi[J], r, maxit=100, atol=1e-6, rtol=1e-6)
        # e = spsolve(self.Bi[J], r)
        e = 0.75 * e
        self.smoothing_time += time.time() - start

        return e
    
    @variantmethod('cg')
    def solve(self, A, F):
        x, info = cg(A, F, maxit=1000, atol=1e-9, rtol=1e-9, returninfo=True)
        # x = spsolve(A, F)
        # print(info)
        return x
    
    @solve.register('pets')
    def solve(self, F, case=1):
        """
        Solve the linear system using direct method.
        """
        case = 1
        ksp = self.A0
        
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

    @solve.register('mg')
    def solve(self, op: PoissonOperator, F):
        # initial set up
        start = time.time()
        self.setup(op)
        self.setup_time += time.time() - start
        self.logger.info(f'Step 4. setup 完成\n')

        k = 0
        r = F
        x = bm.zeros_like(len(F))
        nb = bm.linalg.norm(F)
        err = bm.zeros((self.maxIt, 2), dtype=bm.float64)

        if nb > bm.finfo(float).eps:
            err[0, :] = bm.linalg.norm(r) / nb
        else:
            err[0, :] = bm.linalg.norm(r)

        self.logger.info(f'Step 5. 进入主循环迭代\n')

        start = time.time()
        if self.cycle_type == 'VCYCLE':

            while (bm.max(err[k, :]) > self.tol) & (k <= self.maxIt):
                k = k + 1
                Br = self.vcycle(r)
                x = x + Br

                start0 = time.time()
                r = r - op @ Br
                err[k, 0] = bm.sqrt(bm.abs(Br.T @ r / (x.T @ F)))
                err[k, 1] = bm.linalg.norm(r) / nb
                self.MUL_time += time.time() - start0

                print(
                    f'MG Vcycle iter: {k:2d},   '
                    f'err = {bm.max(err[k, :]):8.4e}'
                )
                
            err = err[:k+1, :]
            itStep = k

        elif self.cycle_type == 'WCYCLE':
            pass
        
        cost = time.time() - start
        self.logger.info(f'Step 6. 程序结束, 开始输出打印结果\n')

        # Output
        print(f"iter: {itStep:2.0f},  "
            f"err = {max(err[-1]):8.4e},  "
            f"level = {self.level},   "
            f"total dof: {self.total_dof:2.0f},   "
            f"coarse dof: {self.coarse_dof:2.0f}\n\n"
            f"total time in coarsest grid: {self.coarse_time}\n"
            f"total time in smoothing: {self.smoothing_time}\n"
            f"total time in MUL of smoothing: {self.MUL_time}\n"
            f"total time: {cost}\n\n"
            f"粗网格总时间占比: {self.coarse_time / cost},  \n"
            f"Smoothing总时间占比: {self.smoothing_time / cost},   \n"
            f"稀疏矩阵@计算总时间占比: {self.MUL_time / cost},  \n",
            f"粗网格和平滑总时间占比: {(self.coarse_time+self.smoothing_time+self.MUL_time) / cost},   \n"
            f"setup时间: {self.setup_time},   \n")

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
        
        return x
    
    def post_process(self, x):
        err = bm.sqrt(bm.mean((self.pde.solution(self.node) - x)**2))
        return err
    

