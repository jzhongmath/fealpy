
from typing import Optional, Tuple, Callable, Union, TypeVar

from ..backend import backend_manager as bm
from ..typing import TensorLike
from ..sparse import SparseTensor, COOTensor, CSRTensor, spdiags
from ..functionspace.space import FunctionSpace

CoefLike = Union[float, int, TensorLike, Callable[..., TensorLike]]
_ST = TypeVar('_ST', bound=SparseTensor)


class StokesDirichletBC():
    """Stokes Dirichlet boundary condition."""
    def __init__(self, space: Tuple[FunctionSpace, ...],
                 gd: Optional[Tuple[CoefLike,...]]=None,
                 *, threshold: Optional[Tuple[CoefLike,...]]=None,
                 method = None):
        self.space = space
        self.gd = gd
        self.threshold = threshold
        self.bctype = 'StokesDirichlet'
        self.method = method
        if isinstance(space, tuple):
            self.gdof = bm.array([i.number_of_global_dofs() for i in space])
            if isinstance(threshold, tuple):
                self.is_boundary_dof = []
                for i in range(len(threshold)):
                    self.is_boundary_dof.append(space[i].is_boundary_dof(threshold[i], method=method))
                self.is_boundary_dof = bm.concatenate(self.is_boundary_dof)
            else:
                self.is_boundary_dof = space[0].is_boundary_dof(threshold, method=method)
                
            self.boundary_dof_index = bm.nonzero(self.is_boundary_dof)[0]
            self.gdof = bm.sum(self.gdof)

    def check_matrix(self, matrix: SparseTensor, /) -> SparseTensor:
        """Check if the input matrix is available for Dirichlet boundary condition.

        Parameters:
            matrix (COOTensor): The left-hand-side matrix of the linear system.

        Raises:
            ValueError: When the layout is not torch.sparse_coo.
            RuntimeError: When the matrix is not coalesced.
            ValueError: When the matrix is not 2-dimensional.
            ValueError: When the matrix is not square.
            ValueError: When the matrix size does not match the gdof of the space.

        Returns:
            Tensor: The input matrix object.
        """
        if not isinstance(matrix, (COOTensor, CSRTensor)):
            raise ValueError('The type of matrix must be COOTensor or CSRTensor.')
        if len(matrix.shape) != 2:
            raise ValueError('The matrix must be a 2-D sparse COO matrix.')
        return matrix

    def check_vector(self, vector: TensorLike, /) -> TensorLike:
        """Check if the input vector is available for Dirichlet boundary conditions.

        Parameters:
            vector (Tensor): The right-hand-side vector of the linear system.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            Tensor: The input vector object.
        """
        if not bm.is_tensor(vector):
            raise ValueError('The type of vector must be a tensor.')
        if vector.ndim not in (1, 2):
            raise ValueError('The vector must be 1-D or 2-D.')
        return vector

    def apply(self, A: SparseTensor, B: SparseTensor, f: TensorLike, g: TensorLike, 
              uh: Optional[TensorLike]=None,
              gd: Optional[CoefLike]=None, *,
              check=True) -> Tuple[TensorLike, TensorLike]:
        """Apply Stokes Dirichlet boundary conditions.
            1. Apply (A, f)
            2. Apply B[:, bd] = 0, g = g - B*u_in
            3. Apply g = g - bm.mean(g)

        Parameters:
            A (SparseTensor): mass sparse matrix.
            B (SparseTensor): speed and pressure.
            f (Tensor): speef vector.
            g (Tensor): pressure vector.

            uh (Tensor | None, optional): The solution uh Tensor. Boundary interpolation\
                will be done on `uh` if given, which is an **in-place** operation.\
                Defaults to None.
            gd (CoefLike | None, optional): The Dirichlet boundary condition.\
                Use the default gd passed in the __init__ if `None`. Default to None.
            check (bool, optional): _description_. Defaults to True.

        Returns:
            out (SparseTensor, Tensor): New adjusted `A` and `f`.
        """
        # 1. Apply (A, f)
        
        f, uh = self.apply_vector(f, A, uh, gd, check=check)
        A = self.apply_matrix(A, check=check)

        # 2. Apply g = g - B*u_in, g = g - bm.mean(g)
        g = g - B.T.matmul(uh[:])
        g = g - bm.mean(g)

        # 3. Apply B[bd, :] = 0
        isDDof = self.is_boundary_dof
        if isinstance(B, CSRTensor):
            B = B.tocoo()
        indices = B.indices
        retain_flag = bm.nonzero(~isDDof[indices[0, :]])[0]
        new_indices = indices[:, retain_flag]
        new_values = B.values[..., retain_flag]
        B = COOTensor(new_indices, new_values, B.sparse_shape)

        return A, B, f, g

    def apply_matrix(self, matrix: _ST, *, check=True) -> _ST:
        """Apply Dirichlet boundary condition to left-hand-size matrix only.

        Parameters:
            matrix (SparseTensor): The original left-hand-size sparse matrix\
                of the linear system.
            check (bool, optional): Whether to check the matrix. Defaults to True.

        Returns:
            SparseTensor: New adjusted left-hand-size matrix.
        """
       
        A = self.check_matrix(matrix) if check else matrix
        isDDof = self.is_boundary_dof
        kwargs = A.values_context()
        bdIdx = bm.zeros(A.shape[0], **kwargs)
        bdIdx = bm.set_at(bdIdx, isDDof.reshape(-1), 1)
        D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        if isinstance(A, COOTensor):
            D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0], format='coo')
        elif isinstance(A, CSRTensor):
            D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0], format='csr')
        #A = D0@A@D0 + D1
        A = self._mul(A, D0) + D1 
        return A

    def apply_vector(self, vector: TensorLike, matrix: SparseTensor,
                     uh: Optional[TensorLike]=None,
                     gd: Optional[CoefLike]=None, *, check=True) -> TensorLike:
        """Apply Dirichlet boundary contition to right-hand-size vector only.

        Parameters:
            vector (TensorLike): The original right-hand-size vector.
            matrix (COOTensor): The original COO/CSR sparse matrix.
            uh (TensorLike | None, optional): The solution uh Tensor. Defuault to None.\
                See `DirichletBC.apply()` for more details.
            gd (CoefLike | None, optional): The Dirichlet boundary condition.\
                Use the default gd passed in the __init__ if `None`. Default to None.
            check (bool, optional): Whether to check the vector. Defaults to True.

        Raises:
            RuntimeError: If gd is `None` and no default gd exists.

        Returns:
            TensorLike: New adjusted right-hand-size vector.
        """
        A = self.check_matrix(matrix) if check else matrix
        f = self.check_vector(vector) if check else vector
        gd = self.gd if gd is None else gd
        
        if gd is None:
            raise RuntimeError("The boundary condition is None.")
        
        suh, sidDDdof = self.space[0].boundary_interpolate(gd=gd,threshold=self.threshold, method=self.method)
        uh = suh
        bd_idx = self.boundary_dof_index
        f = f - A.matmul(uh[:])
        f = bm.set_at(f, bd_idx, uh[bd_idx])
        
        return f, uh

    def _mul(self, A, D0):
        isDDof = self.is_boundary_dof

        if isinstance(A, COOTensor):
            indices = A.indices
            remove_flag = bm.logical_or(
                isDDof[indices[0, :]], isDDof[indices[1, :]]
            )
            retain_flag = bm.logical_not(remove_flag)
            new_indices = indices[:, retain_flag]
            new_values = A.values[..., retain_flag]
            return COOTensor(new_indices, new_values, A.sparse_shape)

        elif isinstance(A, CSRTensor):
            isIDof = bm.logical_not(isDDof)
            crow, col, values = A.crow, A.col, A.values
            indices_context = bm.context(col)
            ZERO = bm.array([0], **indices_context)

            nnz_per_row = crow[1:] - crow[:-1]
            remain_flag = bm.repeat(isIDof, nnz_per_row) & isIDof[col]

            rm_cumsum = bm.concat([ZERO, bm.cumsum(remain_flag, axis=0)], axis=0) # 被保留的非零元素数量累积
            nnz_per_row_new = rm_cumsum[crow[1:]] - rm_cumsum[crow[:-1]]
            nnz_per_row_new = nnz_per_row_new * isIDof

            new_crow = bm.cumsum(bm.concat([ZERO, nnz_per_row_new], axis=0), axis=0)
            new_col = col[remain_flag]
            new_values = values[remain_flag]
            return CSRTensor(new_crow, new_col, new_values, A.sparse_shape)

        else:
            print(f"Warning: Matrix type {type(A)} is not optimized. "
                  "Falling back to memory-intensive sparse multiplication.")
            return D0 @ A @ D0

