import numpy as np
from solver.classes import Settings, Results, Info, QP, DEVICE
import torch

class ADMM_Layer(torch.nn.Module):
    def __init__(self, W, b, l, u, clamp_inds):
        super(ADMM_Layer, self).__init__()
        self.W = W
        self.b = b
        self.l = l
        self.u = u
        self.clamp_inds = clamp_inds

    def forward(self, input):
        input = self.jit_forward(input, self.W, self.b, self.l, self.u, self.clamp_inds[0], self.clamp_inds[1])
        return input

    @torch.jit.script
    def jit_forward(input, W, b, l, u, idx1: int, idx2: int):
        torch.matmul(W, input, out=input)
        input.add_(b)
        input[idx1:idx2].clamp_(l, u)
        return input

class ADMM_NN(torch.nn.Module):
    def __init__(self, N, QP, settings=Settings()):
        super(ADMM_NN, self).__init__()
        torch.set_default_dtype(settings.precision)
        self.QP = QP
        self.settings = settings
        self.clamp_inds = (self.QP.nx + self.QP.nc, self.QP.nx + 2 * self.QP.nc)
        self.N = N
        self.layers = []
        for _ in range(self.N):
            self.layers.append(ADMM_Layer(None, None, self.QP.l, self.QP.u, self.clamp_inds))
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            if idx == self.N-1:
                self.prev_z = x[self.clamp_inds[0]:self.clamp_inds[1]].clone()
            x = layer(x)
        return x

    def update_rho(self, rho):
        device = self.settings.device

        S = torch.diag(torch.tensor(1, device=device) / (rho * self.QP.eigs + torch.tensor(1, device=device))).to(device)
        self.D = self.QP.T @ S @ self.QP.T.T
        rho_inv = torch.tensor(1).to(device=device) / rho
        self.W = torch.cat([
            torch.cat(
                [torch.eye(self.QP.nc, device=device), rho * self.QP.G, -rho * torch.eye(self.QP.nc, device=device)],
                dim=1),
            torch.cat([-self.D @ self.QP.G.T, -rho * self.D @ self.QP.G.T @ self.QP.G, 2 * rho * self.D @ self.QP.G.T],
                      dim=1),
            torch.cat([rho_inv * torch.eye(self.QP.nc, device=device) - self.QP.G @ self.D @ self.QP.G.T,
                       self.QP.G @ (torch.eye(self.QP.nx, device=device) - rho * self.D @ self.QP.G.T @ self.QP.G),
                       rho * 2 * self.QP.G @ self.D @ self.QP.G.T - torch.eye(self.QP.nc, device=device)], dim=1)
        ], dim=0).contiguous()

        Dq = - self.D @ self.QP.q
        self.b = torch.cat([torch.zeros(self.QP.nc).to(self.settings.device), Dq, self.QP.G @ Dq], dim=0).contiguous()

        for layer in self.layers:
            layer.W = self.W
            layer.b = self.b
            layer.l = self.QP.l
            layer.u = self.QP.u


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
              check_interval=25,
              device=DEVICE,
              precision=torch.float32):
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
                                 check_interval=check_interval,
                                 device=device,
                                 precision=precision)

        self.default_rho = torch.tensor(rho, device=self.settings.device,
                                dtype=self.settings.precision).contiguous()
        self.QP = QP(P=P, q=q, G=G, l=l, u=u)
        self.ADMM_nn = ADMM_NN(check_interval, self.QP, self.settings)
        self.ADMM_nn.update_rho(self.default_rho)
        self.x = torch.zeros(self.QP.nx, device=self.settings.device,
                                dtype=self.settings.precision).contiguous()
        self.z = torch.zeros(self.QP.nc, device=self.settings.device,
                                dtype=self.settings.precision).contiguous()
        self.lam = torch.zeros(self.QP.nc, device=self.settings.device,
                                dtype=self.settings.precision).contiguous()
        self.output = torch.cat([self.lam, self.x, self.z]).to(device=self.settings.device, dtype=self.settings.precision).contiguous()

    def update(self, q, u, l=None):
        # QP is updated
        if isinstance(q, np.ndarray):
            q = torch.from_numpy(q)
        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u)
        self.QP.q = q.to(device=self.settings.device, dtype=self.settings.precision)
        self.QP.u = u.to(device=self.settings.device, dtype=self.settings.precision)
        if l is not None:
            if isinstance(l, np.ndarray):
                l = torch.from_numpy(l)
            self.QP.l = l.to(device=self.settings.device, dtype=self.settings.precision)

        # rho is reset
        self.ADMM_nn.update_rho(self.default_rho)

    def solve(self):
        """
        Solve QP Problem
        """
        stng = self.settings
        nx, nc = self.QP.nx, self.QP.nc
        rho = self.default_rho

        for k in range(stng.max_iter // stng.check_interval):
            self.output = self.ADMM_nn(self.output)
            # rho update
            self.lam, self.x, self.z = self.output[:nc], self.output[nc:nc + nx], self.output[nc + nx:nx + 2 * nc]
            primal_res, dual_res, rho = self.compute_residuals(self.QP.G, self.x, self.z, self.ADMM_nn.prev_z, rho)

            if stng.adaptive_rho:
                self.ADMM_nn.update_rho(rho)

            if stng.verbose:
                print('Iter: {}, rho: {:.2e}, res_p: {:.2e}, res_d: {:.2e}'.format(stng.check_interval * k, rho, primal_res, dual_res))

            # check convergence
            if primal_res < stng.eps_abs * np.sqrt(nc) and dual_res < stng.eps_abs * np.sqrt(nx):
                self.update_results(iter=stng.check_interval * k,
                                    status="solved",
                                    pri_res=primal_res,
                                    dua_res=dual_res,
                                    rho_estimate=rho)
                return self.results

        self.lam, self.x, self.z = self.output[:nc], self.output[nc:nc + nx], self.output[nc + nx:nx + 2 * nc]
        primal_res, dual_res, self.rho = self.compute_residuals(self.QP.G, self.x, self.z, self.ADMM_nn.prev_z, rho)

        self.update_results(iter=stng.max_iter, 
                            status="max_iters_reached", 
                            pri_res=primal_res, 
                            dua_res=dual_res, 
                            rho_estimate=self.rho)
        return self.results

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
        self.results.info.obj_val = self.compute_J(H=self.QP.P, g=self.QP.q, x=self.x)
        self.results.info.pri_res = pri_res
        self.results.info.dua_res = dua_res
        self.results.info.rho_estimate = rho_estimate
        self.lam = np.zeros(self.QP.nc)
        self.clear_primal_dual()

    @torch.jit.script
    def compute_residuals(G, x, z, prev_z, rho):
        Gx = G @ x
        primal_res = torch.linalg.vector_norm(Gx - z, ord=torch.inf)
        dual_res = torch.linalg.vector_norm(rho * G.T @ (z - prev_z), ord=torch.inf)

        if primal_res > 10 * dual_res:
            rho = 2 * rho
        elif dual_res > 10 * primal_res:
            rho = rho / 2

        return primal_res, dual_res, rho

    @torch.jit.script
    def compute_J(H, g, x):
        return 0.5*torch.dot(x,torch.matmul(H, x)) + torch.dot(g,x)

    def clear_primal_dual(self):
        """
        Clear primal and dual variables and reset rho index
        """
        self.x = torch.zeros(self.QP.nx, device=self.settings.device,
                             dtype=self.settings.precision).contiguous()
        self.z = torch.zeros(self.QP.nc, device=self.settings.device,
                             dtype=self.settings.precision).contiguous()
        self.lam = torch.zeros(self.QP.nc, device=self.settings.device,
                               dtype=self.settings.precision).contiguous()
        self.output = torch.cat([self.x, self.z, self.lam]).to(device=self.settings.device,
                                                               dtype=self.settings.precision).contiguous()


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

    assert np.allclose(results.x.cpu(), np.array([2.0, -1, 1], dtype=np.float32)), results.x.cpu()
    print(results.x)
    print("Test passed!")


