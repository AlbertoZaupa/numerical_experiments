import numpy as np
from KoopMPC.solver.classes import Settings, Results, Info
from scipy.linalg import eigh

class QPsolver(object):
    def __init__(self):
        super().__init__()

        self.info = Info()
        self.results = Results(info=self.info)

    def setup(self, P, q, G, l, u,
              verbose=False,
              rho=0.1,
              rho_min=1e-6,
              rho_max=1e6,
              adaptive_rho=True,
              adaptive_rho_interval=1,
              adaptive_rho_tolerance=5,
              max_iter=40000,
              eps_abs=1e-3,
              check_interval=25):
        """
        Setup ReLU-QP solver problem of the form

        minimize     1/2 x' * H * x + g' * x
        subject to   l <= A * x <= u

        solver settings can be specified as additional keyword arguments
        """
        self.settings = Settings(verbose=verbose,
                                 rho=rho,
                                 rho_min=rho_min,
                                 rho_max=rho_max,
                                 adaptive_rho=adaptive_rho,
                                 adaptive_rho_interval=adaptive_rho_interval,
                                 adaptive_rho_tolerance=adaptive_rho_tolerance,
                                 max_iter=max_iter,
                                 eps_abs=eps_abs,
                                 check_interval=check_interval)

        self.rho = rho
        self.P = P
        self.q = q
        self.G = G
        self.l = l
        self.u = u
        self.GG = G.T @ G
        self.nx = P.shape[0]
        self.nc = G.shape[0]
        self.setup_hessian_factorization()
        self.x = np.zeros(self.nx)
        self.z = np.zeros(self.nc)
        self.lam = np.zeros(self.nc)

    def setup_hessian_factorization(self):
        L_inv = np.linalg.inv(np.linalg.cholesky(self.P))
        M = L_inv @ self.G.T
        self.eigs, U = eigh(M @ M.T, driver="evd")
        self.T = L_inv.T @ U

    def update(self, q, u, l=None):
        """
        Update ReLU-QP problem arguments
        """
        self.q = q
        if l is not None:
            self.l = l
        self.u = u

    def update_settings(self, **kwargs):
        """
        Update ReLU-QP solver settings

        It is possible to change: 'max_iter', 'eps_abs', 
                                  'verbose', 
                                  'check_interval',
        """
        for key, value in kwargs.items():
            if key in ["max_iter", "eps_ab", "verbose", "check_interval"]:
                setattr(self.settings, key, value)
            elif key in ["rho", "rho_min", "rho_max", "sigma", "adaptive_rho", "adaptive_rho_interval", "adaptive_rho_tolerance"]:
                raise ValueError("Cannot change {} after setup".format(key))
            else:
                raise ValueError("Invalid setting: {}".format(key))

    def solve(self):
        """
        Solve QP Problem
        """
        stng = self.settings
        nx, nc = self.nx, self.nc

        for k in range(1, stng.max_iter + 1):
            self.ADMM_iteration()
            # rho update
            if k % stng.check_interval == 0 and stng.adaptive_rho:
                primal_res, dual_res, self.rho = self.compute_residuals(self.P,
                        self.G, self.q, self.x, self.z, self.lam, self.rho, stng.rho_min, stng.rho_max)

                if stng.verbose:
                    print('Iter: {}, rho: {:.2e}, res_p: {:.2e}, res_d: {:.2e}'.format(k, self.rho, primal_res, dual_res))

                # check convergence
                if primal_res < stng.eps_abs * np.sqrt(nc) and dual_res < stng.eps_abs * np.sqrt(nx):

                    self.update_results(iter=k,
                                        status="solved",
                                        pri_res=primal_res,
                                        dua_res=dual_res,
                                        rho_estimate=self.rho)
                    
                    return self.results

        primal_res, dual_res, rho = self.compute_residuals(self.P, self.G, self.q, self.x, self.z, self.lam, self.rho, stng.rho_min, stng.rho_max)
        self.update_results(iter=stng.max_iter, 
                            status="max_iters_reached", 
                            pri_res=primal_res, 
                            dua_res=dual_res, 
                            rho_estimate=rho)
        return self.results

    def ADMM_iteration(self):
        g = self.q + self.G.T @ (self.lam - self.rho * self.z)
        self.x = - self.T @ np.diag(1 / (self.rho * self.eigs + 1)) @ self.T.T @ g
        #self.x = np.linalg.solve(self.P + self.rho * self.GG, -g)
        self.z = np.clip(self.G @ self.x + self.lam / self.rho, self.l, self.u)
        self.lam = self.lam + self.rho*(self.G @ self.x - self.z)

    def update_results(self, iter=None, 
                       status=None, 
                       pri_res=None, 
                       dua_res=None, 
                       rho_estimate=None):
        """
        Update results and info
        """

        self.results.x = self.x
        self.results.z = self.z
        self.results.lam = self.lam

        self.results.info.iter = iter
        self.results.info.status = status
        self.results.info.obj_val = self.compute_J(H=self.P, g=self.q, x=self.x)
        self.results.info.pri_res = pri_res
        self.results.info.dua_res = dua_res
        self.results.info.rho_estimate = rho_estimate
        self.lam = np.zeros(self.nc)
        self.clear_primal_dual()

    @staticmethod
    def compute_residuals(H, A, g, x, z, lam, rho, rho_min: float, rho_max: float):
        t1 = np.matmul(A, x)
        t2 = np.matmul(H, x)
        t3 = np.matmul(A.T, lam)

        primal_res = np.linalg.norm(t1 - z, ord=np.inf)
        dual_res = np.linalg.norm(t2 + t3 + g, ord=np.inf)
        numerator = np.float64(primal_res) / np.float64(np.max((np.linalg.norm(t1, ord=np.inf), np.linalg.norm(z, ord=np.inf))))
        denom = np.float64(dual_res) / np.float64(np.max((np.linalg.norm(t2, ord=np.inf), np.linalg.norm(t3, ord=np.inf), np.linalg.norm(g, ord=np.inf))))
        rho = np.clip(rho * np.sqrt(numerator / denom), rho_min, rho_max)
        return primal_res, dual_res, rho
    
    @staticmethod
    def compute_J(H, g, x):
        return 0.5*np.dot(x,H @ x) + np.dot(g,x)

    def clear_primal_dual(self):
        """
        Clear primal and dual variables and reset rho index
        """
        self.x = np.zeros(self.nx)
        self.z = np.zeros(self.nc)
        self.lam = np.zeros(self.nc)


if __name__ == "__main__":
    # test on simple QP
    # min 1/2 x' * H * x + g' * x
    # s.t. l <= A * x <= u
    P = np.array([[6, 2, 1], [2, 5, 2], [1, 2, 4.0]], dtype=np.double)
    q = np.array([-8.0, -3, -3], dtype=np.double)
    G = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.double)
    l = np.array([3.0, 0, -10.0, -10, -10], dtype=np.double)
    u = np.array([3.0, 0, np.inf, np.inf, np.inf], dtype=np.double)
        
    qp = QPsolver()
    qp.setup(rho=0.1, P=P, q=q, G=G, l=l, u=u)
    results = qp.solve()

    assert np.allclose(results.x, np.array([2.0, -1, 1], dtype=np.float64)), results
    print(results.x)
    print("Test passed!")


