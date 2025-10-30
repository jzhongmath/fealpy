
__all__ = ['LinearMeshCFEDof']

from typing import Union, Generic, TypeVar

from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..mesh.mesh_base import Mesh


_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, TensorLike]
_S = slice(None)


class LinearMeshCFEDof(Generic[_MT]):
    def __init__(self, mesh: _MT, p: int):
        TD = mesh.top_dimension()
        self.mesh = mesh
        self.p = p
        self.multiIndex = mesh.multi_index_matrix(p, TD)
   
    def is_boundary_dof(self, threshold=None, method=None):
        TD = self.mesh.top_dimension()
        gdof = self.number_of_global_dofs()
        if bm.is_tensor(threshold):
            index = threshold
            if (index.dtype == bm.bool) and (len(index) == gdof):
                return index
            else:
                raise ValueError(f"Unknown threshold: {threshold}")
        else:
            if (method == 'centroid') | (method is None):
                index = self.mesh.boundary_face_index()
                isBdDof = bm.zeros(gdof, dtype=bm.bool, device=bm.get_device(self.mesh))
                
                if self.mesh.meshtype in ['prism', 'tensorprism']:
                    if callable(threshold):
                        tface = self.mesh.tface[index[0]]
                        qface = self.mesh.qface[index[1]]
                        bc0 = bm.barycenter(tface, self.mesh.node)
                        bc1 = bm.barycenter(qface, self.mesh.node)
                        flag0 = threshold(bc0)
                        index0 = index[0][flag0]
                        flag1 = threshold(bc1)
                        index1 = index[1][flag1]      
                    
                    tface2dof = self.mesh.tri_to_ipoint(self.p, index0)
                    qface2dof = self.mesh.quad_to_ipoint(self.p, index1)
                    isBdDof = bm.set_at(isBdDof, tface2dof, True)
                    isBdDof = bm.set_at(isBdDof, qface2dof, True)
                else:
                    if callable(threshold):
                        bc = self.mesh.entity_barycenter(TD-1, index=index)
                        flag = threshold(bc)
                        index = index[flag]
                    face2dof = self.face_to_dof(index=index) # 只获取指定的面的自由度信息
                    isBdDof = bm.set_at(isBdDof, face2dof, True)

            elif method == 'interp':
                index = self.mesh.boundary_face_index()
                if self.mesh.meshtype in ['prism', 'tensorprism']:
                    tface2dof = self.mesh.tri_to_ipoint(self.p, index[0])
                    qface2dof = self.mesh.quad_to_ipoint(self.p, index[1])
                    index_dof = bm.concat([tface2dof.flatten(), qface2dof.flatten()], axis=0)
                else:
                    face2dof = self.face_to_dof(index=index) # 只获取指定的面的自由度信息
                    index_dof = face2dof.flatten()

                if callable(threshold):
                    ##TODO, index_dof加插值点函数里
                    # import ipdb;ipdb.set_trace()
                    ipoint = self.mesh.interpolation_points(p=self.p)[index_dof]
                    flag = threshold(ipoint)
                    index_dof = index_dof[flag]
                isBdDof = bm.zeros(gdof, dtype=bm.bool, device=bm.get_device(self.mesh))
                isBdDof = bm.set_at(isBdDof, index_dof, True)
            else:
                raise ValueError(f"Unknown method: {method}")
        return isBdDof

    def entity_to_dof(self, etype: int, index: Index=_S):
        TD = self.mesh.top_dimension()
        if etype == TD:
            return self.cell_to_dof(index)
        elif etype == TD-1:
            return self.face_to_dof(index)
        elif etype == 1:
            return self.edge_to_dof(index)
        else:
            raise ValueError(f"Unknown entity type: {etype}")

    def edge_to_dof(self, index: Index=_S):
        return self.mesh.edge_to_ipoint(self.p, index=index)

    def face_to_dof(self, index: Index=_S):
        return self.mesh.face_to_ipoint(self.p, index=index)

    def cell_to_dof(self, index: Index=_S):
        return self.mesh.cell_to_ipoint(self.p, index=index)

    def interpolation_points(self, index: Index=_S) -> TensorLike:
        return self.mesh.interpolation_points(self.p, index=index)

    def number_of_global_dofs(self) -> int:
        return self.mesh.number_of_global_ipoints(self.p)

    def number_of_local_dofs(self, doftype='cell') -> int:
        return self.mesh.number_of_local_ipoints(self.p, iptype=doftype)
    
class LinearMeshDFEDof(Generic[_MT]):
    def __init__(self, mesh: _MT, p: int):
        TD = mesh.top_dimension()
        self.mesh = mesh
        self.p = p
        if p > 0:
            self.multiIndex = mesh.multi_index_matrix(p, TD)
        else:
            TD = mesh.top_dimension()
            self.multiIndex = bm.array((TD+1)*(0,), dtype=mesh.itype)
        self.cell2dof = self.cell_to_dof()

    def entity_to_dof(self, etype: int, index: Index=_S):
        TD = self.mesh.top_dimension()
        if etype == TD:
            return self.cell_to_dof(index)
        else:
            raise ValueError(f"Unknown entity type: {etype}")

    def cell_to_dof(self, index : Index=_S) -> TensorLike:
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs()
        cell2dof = bm.arange(NC*ldof).reshape(NC, ldof)

        return cell2dof[index]

    def number_of_global_dofs(self):
        NC = self.mesh.number_of_cells()
        ldof = self.number_of_local_dofs()
        gdof = ldof*NC
        
        return gdof
    
    def number_of_local_dofs(self, doftype='cell') -> int:
        return self.mesh.number_of_local_ipoints(self.p, iptype=doftype)

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')
        GD = mesh.geo_dimension()

        if p == 0:
            return mesh.entity_barycenter('cell')

        if p == 1:
            return node[cell].reshape(-1, GD)

        w = self.multiIndex/p
        ipoint = bm.einsum('ij, kj...->ki...', w, node[cell]).reshape(-1, GD)
        
        return ipoint