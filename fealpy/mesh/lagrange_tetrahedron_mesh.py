from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .utils import simplex_gdof, simplex_ldof 
from .mesh_base import HomogeneousMesh, estr2dim
from .tetrahedron_mesh import TetrahedronMesh


from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .utils import simplex_gdof, simplex_ldof 
from .mesh_base import HomogeneousMesh, estr2dim
from .tetrahedron_mesh import TetrahedronMesh


class LagrangeTetrahedronMesh(HomogeneousMesh):
    """
    
    Parameters:
        node(TensorLike): the coordinates of the nodes.

        cell(TensorLike): the connectivity of the cells.

        p(int, optional): the order of the Lagrange element. If p is None,
            it will be computed from cell.shape[-1].

        boundary(Boundary, optional): the boundary object of the mesh.

        surface(Surface, optional): the surface object contained the mesh.

    Attributes:

    Methods:

    Notes:

    Todos:
    """
    def __init__(self, 
                 node: TensorLike, 
                 cell: TensorLike, 
                 p: Optional[int] = None, 
                 boundary=None, 
                 surface=None):
    
        super().__init__(TD=2, itype=cell.dtype, ftype=node.dtype)

    