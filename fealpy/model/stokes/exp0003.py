from ...backend import bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher3d
from typing import Sequence


class Exp0002(BoxMesher3d):
    """
    Analytic solution to a 3D flow problem with zero divergence:

        -μ Δu + ∇p = f,  in Ω = [0,1]^3,
               ∇⋅u = 0,  in Ω = [0,1]^3,
                 u = g,  on ∂Ω.

    With manufactured solution:
        u1 =   sin(πx)cos(πy)cos(πz),
        u2 =   cos(πx)sin(πy)cos(πz),
        u3 = -2cos(πx)cos(πy)sin(πz).

    We have ∇⋅u = 0. Take 
            p(x, y, z) = sin(2πx) + cos(2πy) + sin(2πz),
        and
            ∇p = (2πcos(2πx), -2πsin(2πy), 2πcos(2πz)),

        we have
            f = -Δu + ∇p = 3π^2u + ∇p.

    The body force f is computed to satisfy -μ Δu + ∇p = f with μ = 1.
    """
    def __init__(self, option: dict = {}):
        self.box = [0, 1, 0, 1, 0, 1]
        self.mu = bm.tensor(option.get("mu", 1.0))
        super().__init__(box=self.box)

    def geo_dimension(self) -> int:
        return 3

    def domain(self) -> Sequence[float]:
        return self.box
    
    @cartesian
    def is_velocity_boundary(self, p: TensorLike, dim=3) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        atol = 1e-12
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        if dim == 2:
            return (
                (bm.abs(x - 0.) < atol) | (bm.abs(x - 1.) < atol) |
                (bm.abs(y - 0.) < atol) | (bm.abs(y - 1.) < atol)
            )
        
        return (
            (bm.abs(x - 0.) < atol) | (bm.abs(x - 1.) < atol) |
            (bm.abs(y - 0.) < atol) | (bm.abs(y - 1.) < atol) |
            (bm.abs(z - 0.) < atol) | (bm.abs(z - 1.) < atol)
        )
    
    @cartesian
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed velocity on boundary, if needed explicitly."""

        return self.velocity(p)
    
    @cartesian
    def velocity(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        result = bm.zeros(p.shape, dtype=bm.float64)

        result[...,0] = bm.sin(bm.pi*x) * bm.cos(bm.pi*y) * bm.cos(bm.pi * z)
        result[...,1] = bm.cos(bm.pi*x) * bm.sin(bm.pi*y) * bm.cos(bm.pi * z)
        result[...,2] = -2*bm.cos(bm.pi*x) * bm.cos(bm.pi*y) * bm.sin(bm.pi*z)

        return result

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        """Optional: prescribed pressure on boundary (usually for stability)."""
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        result = bm.sin(2*bm.pi*x) + bm.cos(2*bm.pi*y) + bm.sin(2*bm.pi*z)

        return result

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        result = bm.zeros(p.shape, dtype=bm.float64)
        result[...,0] =  2*bm.pi * bm.cos(2*bm.pi*x)
        result[...,1] = -2*bm.pi * bm.sin(2*bm.pi*y)
        result[...,2] =  2*bm.pi * bm.cos(2*bm.pi*z)

        return result + 3*bm.pi**2*self.velocity(p)

    @cartesian
    def source0(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        return 2*bm.pi * bm.cos(2*bm.pi*x) + 3*bm.pi**2*bm.sin(bm.pi*x) * bm.cos(bm.pi*y) * bm.cos(bm.pi * z)

    @cartesian
    def source1(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        return -2*bm.pi * bm.sin(2*bm.pi*y) + 3*bm.pi**2*bm.cos(bm.pi*x) * bm.sin(bm.pi*y) * bm.cos(bm.pi * z)

    @cartesian
    def source2(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        return 2*bm.pi * bm.cos(2*bm.pi*z) + 3*bm.pi**2*(-2)*bm.cos(bm.pi*x) * bm.cos(bm.pi*y) * bm.sin(bm.pi*z)

