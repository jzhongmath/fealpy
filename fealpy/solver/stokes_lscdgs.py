
from ..backend import bm
from .mg import mg

from scipy.sparse.linalg import spsolve_triangular
from scipy.sparse.linalg import cg, tfqmr
from joblib import Parallel, delayed
import pyamg
from scipy.sparse import diags

# from ..solver import spsolve, spsolve_triangular

import time 

class StokesLSCDGS():
    def __init__(self, 
        # u,p,f,g,A,B,
        auxMat,
        smootherOpt
    ):  
        self.set_up(auxMat, smootherOpt)
        # self.run(u,p,f,g,A,B)
    
    def set_up(self, auxMat, smootherOpt):
        self.smoothingstep = smootherOpt.get('smoothingstep', 2)
        self.smoothingSp = smootherOpt.get('smoothingSp', 'GS')
        self.smoothingbarSp = smootherOpt.get('smoothingbarSp', 'GS')
        self.smoothingbarSpPara = smootherOpt.get('smoothingbarSpPara', 1)
        if (self.smoothingbarSp == 'VCYCLE') or (self.smoothingSp == 'VCYCLE'):
            self.optionmg = {
                'solvermaxit': 1,
                'solver': 'VCYCLE',
                'smoothingstep': 2,
                'printlevel': 0,
                'setupflag': 0
            }

        self.Bt = auxMat.get('Bt')
        self.BBt = auxMat.get('BBt')
        self.BABt = auxMat.get('BABt')
        self.Su = auxMat.get('Su')
        self.Sp = auxMat.get('Sp')
        self.Spt = auxMat.get('Spt')
        self.DSp = auxMat.get('DSp')

        # self.elem = elem
        # self.Ai = Ai
        # self.Si = Si
        # self.SSi = SSi
        # self.Res = Res
        # self.Pro = Pro

    def run(self, u,p,f,g,A,B,mul_time):
        for _ in range(self.smoothingstep):
            # Step 1: relax Momentum eqns
            
            # u = u + self.Su / (f - self.Bt @ p - A @ u)
            r = (f - self.Bt @ p - A @ u)
            start = time.time()
            # u = u + pyamg.ruge_stuben_solver(self.Su).solve(r, maxiter=1)
            # u = u + spsolve_triangular(self.Su, r, lower=True)
            u = u + tfqmr(self.Su, r, maxiter=3)[0]
            # u = u + 0.5*r / self.Su
            mul_time += time.time() - start
            # Step 2: relax transformed Continuity eqns
            rp = g - B @ u
                       
            if self.smoothingSp == 'SGS':
                b0 = self.DSp * spsolve_triangular(self.Sp, rp, lower=True)
                dq = spsolve_triangular(self.Spt, b0, lower=False)
                # dq = self.Spt / (self.DSp @ (self.Sp / rp))
            elif self.smoothingSp == 'GS':
                start = time.time()
                # mg = pyamg.ruge_stuben_solver(self.Sp)
                # dq = mg.solve(rp, maxiter=3)
                dq = tfqmr(self.Sp, rp, rtol=1e-1)[0]
                # dq = spsolve_triangular(self.Sp, rp, lower=True)
                mul_time += time.time() - start
            elif self.smoothingSp == 'VCYCLE':
                pass

            # Step 3: transform the correction back lower=Falseto the original variables
            u = u + self.Bt @ dq
            dq = self.BABt @ dq
            
            # dq = dq - bm.mean(dq)
            if self.smoothingbarSp == 'SGS':
                b1 = self.DSp * spsolve_triangular(self.Spt, dq, lower=False)
                # dp = spsolve_triangular(self.Sp, b1, lower=True)
                dp = spsolve_triangular(self.Sp, b1, lower=True)
                # dp = self.Sp / (self.DSp @ (self.Spt / dq))
            elif self.smoothingbarSp == 'GS':
                # dp = self.Spt / dq
                start = time.time()
                # dp = spsolve_triangular(self.Spt, dq, lower=False)
                # dp = pyamg.ruge_stuben_solver(self.Spt).solve(dq, maxiter=1)
                dp = tfqmr(self.Spt, dq, maxiter=3)[0]
                mul_time += time.time() - start
            elif self.smoothingbarSp == 'VCYCLE':
                pass
            
            p = p - self.smoothingbarSpPara*dp
        
        return u, p, mul_time
    