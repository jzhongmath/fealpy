
from typing import Optional, Tuple, Callable, Union

from ..backend import backend_manager as bm
from ..typing import TensorLike
from ..sparse import COOTensor
from ..functionspace import LagrangeFESpace
from ..functionspace.space import FunctionSpace

from .form import Form

CoefLike = Union[float, int, TensorLike, Callable[..., TensorLike]]

class LinearOperator:
    """Generic linear operator interface."""

    def __matmul__(self, x: TensorLike) -> TensorLike:
        raise NotImplementedError

    def __add__(self, other: "LinearOperator") -> "SumLinearOperator":

        return SumLinearOperator(self, other)


class KronDirichletBCOperator(LinearOperator):
    """Dirichlet boundary condition operator."""
    def __init__(self, form0: Form, form1: Form, 
                 form2: Form, form3: Form, space: LagrangeFESpace,
                 gd: Optional[CoefLike]=None,
                 *, threshold: Optional[Callable]=None, 
                 isDDof=None,
                 left:bool=True):
        
        self.form0 = form0
        self.form1 = form1
        self.form2 = form2
        self.form3 = form3
        self.gd = gd
        self.space = space
        if isDDof is None:
            isDDof = space.is_boundary_dof(threshold=threshold) # on the same device as space
            self.is_boundary_dof = isDDof
        else :
            self.is_boundary_dof = isDDof
        self.boundary_dof_index = bm.nonzero(isDDof)[0]
        self.shape0 = form0.shape 
        self.shape1 = form1.shape

    def init_solution(self):
        """
        Generate the init solution with correct Dirichlet boundary
        condition.

        Returns:
            u (TensorLike): the init solution.
        TODO:
            1. deal with device
        """
        uh = bm.zeros(self.shape0[0] * self.shape1[0], dtype=self.space.ftype)
        self.space.boundary_interpolate(self.gd, uh,
                threshold=self.is_boundary_dof)
        return uh

    def apply(self, F, uh):
        X = bm.reshape(uh, (self.shape0[0], self.shape1[0]))
        AXT0 = bm.zeros_like(X, dtype=self.space.ftype)
        AXT1 = bm.zeros_like(X, dtype=self.space.ftype)
        for i in range(self.shape1[0]):
            AXT0[:, i] = self.form0 @ X[:, i]
            AXT1[:, i] = self.form2 @ X[:, i]
        AXT0 = AXT0.T
        AXT1 = AXT1.T
        # AXT0 = (self.form0 @ X).T 
        # AXT1 = (self.form2 @ X).T
        # M = bm.zeros_like(X.T, dtype=self.space.ftype)
        # for i in range(self.shape0[0]):
        #     M[:, i] = self.form1 @ AXT0[:, i] + self.form3 @ AXT1[:, i]
        # F = F - M.T.reshape(-1)
        F = F - (self.form1 @ AXT0).T.reshape(-1) - (self.form3 @ AXT1).T.reshape(-1)
        F = bm.set_at(F, self.is_boundary_dof, uh[self.is_boundary_dof])
        return F

    def __matmul__(self, u: TensorLike):
        """Apply the dirichlet boundary condition on the matrix-vetor multiply.

        Parameters:
            u (TensorLike): the input vector.

        Returns:
            v (TensorLike): the result of matrix-vector multiply.

        TODO:
            1. support for v.shape[0] != u.shape[0]
        """
        v = bm.copy(u) 
        val = v[self.is_boundary_dof]
        bm.set_at(v, self.is_boundary_dof, 0.0)
        X = bm.reshape(v, (self.shape0[0], self.shape1[0]))
        AXT0 = bm.zeros_like(X, dtype=self.space.ftype)
        AXT1 = bm.zeros_like(X, dtype=self.space.ftype)
        for i in range(self.shape1[0]):
            AXT0[:, i] = self.form0 @ X[:, i]
            AXT1[:, i] = self.form2 @ X[:, i]
        AXT0 = AXT0.T
        AXT1 = AXT1.T
        M = bm.zeros_like(X.T, dtype=self.space.ftype)
        for i in range(self.shape0[0]):
            M[:, i] = self.form1 @ AXT0[:, i] + self.form3 @ AXT1[:, i]
        v = M.T.reshape(-1) 
        bm.set_at(v, self.is_boundary_dof, val) 
        return v

class SumLinearOperator(LinearOperator):
    """Sum of two linear operators."""
    def __init__(self, A: LinearOperator, B: LinearOperator):
        self.A = A
        self.B = B

    def __matmul__(self, x: TensorLike) -> TensorLike:
        return self.A @ x + self.B @ x
    
