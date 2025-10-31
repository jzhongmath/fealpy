from ..backend import bm
from ..mesh import TriangleMesh


def transferP1red(mesh: TriangleMesh, level:int, threshold:None):
        Pro_p = [None]*(level-1)
        for i in range(level-1):
            idof0 = threshold(mesh.entity('node'))
            P = mesh.uniform_refine(n=1, returnim=True)
            idof1 = threshold(mesh.entity('node'))
            Pro_p[i] = P[idof0, idof1]

        return Pro_p