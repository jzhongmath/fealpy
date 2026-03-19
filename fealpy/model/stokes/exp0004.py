from ...backend import bm
from ...decorator import cartesian
from ...typing import TensorLike
from ...mesher import BoxMesher3d
from typing import Sequence


class Exp0004(BoxMesher3d):
    """
    Analytic solution to a 3D flow problem with zero divergence:

        -μ Δu + ∇p = f,  in Ω = [0,1]^3,
               ∇⋅u = 0,  in Ω = [0,1]^3,
                 u = g,  on ∂Ω.

    With manufactured solution:
        u1 =   sin^2(πx)sin(2πy)sin(2πz),
        u2 =  -sin(2πx)sin^2(πy)sin(2πz),
        u3 =   0.

        and 
            Δu1 = 2π^2(3cos(2πx)-2)sin(2πy)sin(2πz),

            Δu2 = -2π^2(3cos(2πy)-2)sin(2πx)sin(2πz),

            Δu3 = 0.

    We have ∇⋅u = 0. Take 
            p(x, y, z) = cos(2πx)cos(2πy)cos(2πz),
        and
            ∇p = (
                  -2πsin(2πx)cos(2πy)cos(2πz), 
                  -2πcos(2πx)sin(2πy)cos(2πz), 
                  -2πcos(2πx)cos(2πy)sin(2πz)
                )

        we have
            f1 = -2π^2(3cos(2πx)-2)sin(2πy)sin(2πz) - 2πsin(2πx)cos(2πy)cos(2πz),
            f2 = 2π^2(3cos(2πy)-2)sin(2πx)sin(2πz) - 2πcos(2πx)sin(2πy)cos(2πz),
            f3 = -2πcos(2πx)cos(2πy)sin(2πz).

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

        # u1 = sin²(πx) * sin(2πy) * sin(2πz)
        result[..., 0] = (bm.sin(pi * x) ** 2) * bm.sin(2 * pi * y) * bm.sin(2 * pi * z)

        # u2 = -sin(2πx) * sin²(πy) * sin(2πz)
        result[..., 1] = -bm.sin(2 * pi * x) * (bm.sin(pi * y) ** 2) * bm.sin(2 * pi * z)

        # u3 = 0
        result[..., 2] = 0.0

        return result

    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = bm.pi

        # p = cos(2πx) cos(2πy) cos(2πz)
        return bm.cos(2 * pi * x) * bm.cos(2 * pi * y) * bm.cos(2 * pi * z)

    @cartesian
    def source0(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = bm.pi
        pi2 = pi ** 2

        # Δu1 = 2π² (3cos(2πx) - 2) sin(2πy) sin(2πz)
        lap_u1 = 2 * pi2 * (3 * bm.cos(2 * pi * x) - 2) * bm.sin(2 * pi * y) * bm.sin(2 * pi * z)

        # ∂p/∂x = -2π sin(2πx) cos(2πy) cos(2πz)
        grad_p_x = -2 * pi * bm.sin(2 * pi * x) * bm.cos(2 * pi * y) * bm.cos(2 * pi * z)

        # f1 = -Δu1 + ∂p/∂x
        return -lap_u1 + grad_p_x

    @cartesian
    def source1(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = bm.pi
        pi2 = pi ** 2

        # Δu2 = -2π² (3cos(2πy) - 2) sin(2πx) sin(2πz)
        lap_u2 = -2 * pi2 * (3 * bm.cos(2 * pi * y) - 2) * bm.sin(2 * pi * x) * bm.sin(2 * pi * z)

        # ∂p/∂y = -2π cos(2πx) sin(2πy) cos(2πz)
        grad_p_y = -2 * pi * bm.cos(2 * pi * x) * bm.sin(2 * pi * y) * bm.cos(2 * pi * z)

        # f2 = -Δu2 + ∂p/∂y
        return -lap_u2 + grad_p_y

    @cartesian
    def source2(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = bm.pi

        # ∂p/∂z = -2π cos(2πx) cos(2πy) sin(2πz)
        grad_p_z = -2 * pi * bm.cos(2 * pi * x) * bm.cos(2 * pi * y) * bm.sin(2 * pi * z)

        # u3 = 0 ⇒ Δu3 = 0 ⇒ f3 = -0 + ∂p/∂z
        return grad_p_z

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Vector-valued body force f = -Δu + ∇p"""
        f1 = self.source0(p)
        f2 = self.source1(p)
        f3 = self.source2(p)
        return bm.stack([f1, f2, f3], axis=-1)
