from ...backend import bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher3d
from typing import Sequence


class Exp0005(BoxMesher3d):
    """
    Analytic solution to a 3D flow problem with zero divergence:

        -μ Δu + ∇p = f,  in Ω = [0,1]^3,
               ∇⋅u = 0,  in Ω = [0,1]^3,
                 u = g,  on ∂Ω.

    With manufactured solution:
        u1 =   3πsin(2πx)cos(3πy)sin(πz),
        u2 =  -2πcos(2πx)sin(3πy)sin(πz) + 3πcos(πx)sin(2πy)cos(3πz),
        u3 =  -2πcos(πx)cos(2πy)sin(3πz).

        and 
            Δu = -14π^2u.

    We have ∇⋅u = 0. Take 
            p(x, y, z) = sin(πx)cos(2πy)sin(3πz),
        and
            ∇p = (
                  πcos(πx)cos(2πy)sin(3πz), 
                -2πsin(πx)sin(2πy)sin(3πz), 
                 3πsin(πx)cos(2πy)cos(3πz)
                ).

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
        pi = bm.pi
        result = bm.zeros(p.shape, dtype=bm.float64)

        result[..., 0] = 3 * pi * bm.sin(2 * pi * x) * bm.cos(3 * pi * y) * bm.sin(pi * z)

        result[..., 1] = -2 * pi * bm.cos(2 * pi * x) * bm.sin(3 * pi * y) * bm.sin(pi * z) \
                + 3 * pi * bm.cos(pi * x) * bm.sin(2 * pi * y) * bm.cos(3 * pi * z)

        result[..., 2] = -2 * pi * bm.cos(pi * x) * bm.cos(2 * pi * y) * bm.sin(3 * pi * z)

        return result

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = bm.pi

        return bm.sin(pi * x) * bm.cos(2 * pi * y) * bm.sin(3 * pi * z)
    
    @cartesian
    def source0(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = bm.pi
        pi2 = pi ** 2

        lap_u1 = -14 * pi2 * 3 * pi * bm.sin(2 * pi * x) * bm.cos(3 * pi * y) * bm.sin(pi * z)
        grad_p_x = pi * bm.cos(pi * x) * bm.cos(2 * pi * y) * bm.sin(3 * pi * z)

        return -lap_u1 + grad_p_x

    @cartesian
    def source1(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = bm.pi
        pi2 = pi ** 2

        lap_u2 = -14 * pi2 * (-2 * pi * bm.cos(2 * pi * x) * bm.sin(3 * pi * y) * bm.sin(pi * z) \
                + 3 * pi * bm.cos(pi * x) * bm.sin(2 * pi * y) * bm.cos(3 * pi * z))
        grad_p_y = -2 * pi * bm.sin(pi * x) * bm.sin(2 * pi * y) * bm.sin(3 * pi * z)

        return -lap_u2 + grad_p_y

    @cartesian
    def source2(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = bm.pi

        lap_u3 = -14 * pi**2 * (-2 * pi * bm.cos(pi * x) * bm.cos(2 * pi * y) * bm.sin(3 * pi * z))
        grad_p_z = 3 * pi * bm.sin(pi * x) * bm.cos(2 * pi * y) * bm.cos(3 * pi * z)

        return -lap_u3 + grad_p_z

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Vector-valued body force f = -Δu + ∇p"""
        f1 = self.source0(p)
        f2 = self.source1(p)
        f3 = self.source2(p)
        return bm.stack([f1, f2, f3], axis=-1)
