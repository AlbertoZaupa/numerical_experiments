import torch
import numpy as np
from scipy.linalg import eigh

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

class QP(object):
    def __init__(self, P: torch.tensor or np.ndarray,
                 q: torch.tensor or np.ndarray,
                 G: torch.tensor or np.ndarray,
                 l: torch.tensor or np.ndarray, u: torch.tensor or np.ndarray,
                 device=torch.device(DEVICE), precision=torch.float32):

        # convert to torch tensors if it's numpy array
        if isinstance(P, np.ndarray):
            P = torch.from_numpy(P)
        if isinstance(q, np.ndarray):
            q = torch.from_numpy(q)
        if isinstance(G, np.ndarray):
            G = torch.from_numpy(G)
        if isinstance(l, np.ndarray):
            l = torch.from_numpy(l)
        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u)

        self.P = P.to(device=device, dtype=precision).contiguous()
        self.q = q.to(device=device, dtype=precision).contiguous()
        self.G = G.to(device=device, dtype=precision).contiguous()
        self.l = l.to(device=device, dtype=precision).contiguous()
        self.u = u.to(device=device, dtype=precision).contiguous()
        L = np.linalg.cholesky(P)
        self.eigs, self.T = self.setup_hessian_factorization(L=L, G=G, device=device, precision=precision)

        self.nx = P.shape[0]  # number of decision variables
        self.nc = G.shape[0]  # number of constraints

    def setup_hessian_factorization(self, L, G, device, precision):
        L_inv = torch.linalg.inv(torch.from_numpy(L))
        M = L_inv @ G.T
        eigs, U = eigh((M @ M.T).numpy(), driver="evd")
        eigs, U = torch.from_numpy(eigs).to(device=device, dtype=precision).contiguous(), torch.from_numpy(U)
        T = (L_inv.T @ U).to(device=device, dtype=precision).contiguous()
        return eigs, T

class Settings(object):
    def __init__(self, verbose=False,
                        rho=0.1,
                        rho_min=1e-6,
                        rho_max=1e6,
                        adaptive_rho=True,
                        adaptive_rho_interval=1,
                        adaptive_rho_tolerance=5,
                        max_iter=4000,
                        eps_abs=1e-3,
                        eq_tol=1e-6,
                        check_interval=25,
                        device=DEVICE,
                        precision=torch.float32):
        
        self.verbose = verbose
        self.rho = rho
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.adaptive_rho = adaptive_rho
        self.adaptive_rho_interval = adaptive_rho_interval
        self.adaptive_rho_tolerance = adaptive_rho_tolerance
        self.max_iter = max_iter
        self.eps_abs = eps_abs
        self.eq_tol = eq_tol
        self.check_interval = check_interval
        self.device = device
        self.precision = precision

class Info(object):
    def __init__(self, iter=None, 
                        status=None, 
                        obj_val=None,
                        pri_res=None,
                        dua_res=None,
                        rho_estimate=None,
                 ):
        self.iter = iter
        self.status = status
        self.obj_val = obj_val
        self.pri_res = pri_res
        self.dua_res = dua_res
        self.rho_estimate = rho_estimate


class Results(object):
    def __init__(self, x=None, z=None, lam=None, info: Info=None):
        self.x = x
        self.z = z
        self.lam = lam
        self.info = info

