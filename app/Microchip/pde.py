from typing import Sequence
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike
from .chip_mesher import ChipMesher
import sympy as sp

class Pde:
    def __init__(self, options: dict = {}):
        self.options = options
        self.eps = 1e-10
        self.mu = 1e-3
        self.rho = 1.0

    def get_dimension(self) -> int: 
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.options['box']
    
    def init_mesh(self):
        self.box = self.options['box']
        mesher = ChipMesher(options=self.options)
        self.centers = mesher.centers
        self.mesh = mesher.mesh
        return self.mesh
    
    @cartesian
    def velocity_dirichlet(self, p:TensorLike) -> TensorLike:
        inlet = self.inlet_velocity(p)
        outlet = self.outlet_velocity(p)
        is_inlet = self.is_inlet_boundary(p)
        is_outlet = self.is_outlet_boundary(p)
        
        result = bm.zeros_like(p, dtype=p.dtype)
        result[is_inlet] = inlet[is_inlet]
        result[is_outlet] = outlet[is_outlet]
        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        return self.outlet_pressure(p)

    @cartesian
    def inlet_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 1.2 * y * (0.41 - y)/(0.41**2)
        result[..., 1] = bm.array(0.0)
        return result
    
    @cartesian
    def outlet_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 1.2 * y * (0.41 - y)/(0.41**2)
        result[..., 1] = bm.array(0.0)
        return result
    
    @cartesian
    def outlet_pressure(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of pressure."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.array(0.0)
        return result
    
    @cartesian
    def wall_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity on wall."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = bm.array(0.0)
        result[..., 1] = bm.array(0.0)
        return result
    
    @cartesian
    def obstacle_velocity(self, p: TensorLike) -> TensorLike:
        """Compute exact solution of velocity on obstacle."""
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = bm.array(0.0)
        result[..., 1] = bm.array(0.0)
        return result
    
    @cartesian
    def pressure(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        result = bm.array(0.0)
        return result
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        x = p[..., 0]
        y = p[..., 1]
        result = bm.zeros(p.shape, dtype=bm.float64)
        result[..., 0] = 0.0
        result[..., 1] = 0.0
        return result
    
    @cartesian
    def is_velocity_boundary(self, p):
        return None
    
    @cartesian
    def is_pressure_boundary(self, p):
        is_out = self.is_outlet_boundary(p)
        return is_out
    
    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        # 检查是否接近 x=±1 或 y=±1
        on_boundary = (bm.abs(x - self.box[0]) < atol)
        return on_boundary

    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where pressure is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        on_boundary = (bm.abs(x - self.box[1]) < atol)
        return on_boundary
    
    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        atol = 1e-12
        on_boundary = (
            (bm.abs(y - self.box[2]) < atol) | (bm.abs(y - self.box[3]) < atol))
        return on_boundary
    
    @cartesian
    def is_obstacle_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point where velocity is defined is on boundary."""
        x = p[..., 0]
        y = p[..., 1]
        radius = self.options['radius']
        atol = 1e-12
        # 检查是否接近圆的边界
        on_boundary = bm.zeros_like(x, dtype=bool)
        for center in self.centers:
            cx, cy = center
            on_boundary |= bm.abs((x - cx)**2 + (y - cy)**2 - radius**2) < atol
        return on_boundary
        