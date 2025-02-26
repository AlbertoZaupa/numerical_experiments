import torch
from torch.utils.data import Dataset
from pykoopman.common.examples import vdp_osc, rk4

class VdPDataset(Dataset):
    def __init__(self, N, n_traj, n_int, dT):
        super(VdPDataset, self).__init__()
        self.num_samples = (n_int - 1 + 1 - N) * n_traj
        self.N = N
        self.dT = dT
        self.n_states = 2
        self.n_inputs = 1

        # Uniform random distributed forcing in [-1, 1]
        u = 2 * torch.rand((n_int, n_traj)) - 1

        # Uniform distribution of initial conditions
        # 2 is the number of state components
        x = 2 * torch.rand([self.n_states, n_traj]) - 1

        # Init
        X = torch.zeros((self.n_states * n_int, n_traj))
        U = torch.zeros((self.n_inputs * n_int, n_traj))

        # Integrate
        for step in range(n_int):
            y = rk4(0, x, u[step, :], dT, vdp_osc)
            X[self.n_states * step: self.n_states * (step+1), :] = x
            U[self.n_inputs * step: self.n_inputs * (step+1), :] = u[step, :]
            x = y

        # 1 is the size of the control vector
        self.U = sliding_window_2d(U[:-self.n_inputs, :], 1, self.N)
        self.target = sliding_window_2d(X[self.n_states:, :], self.n_states, self.N)
        self.x0 = sliding_window_2d(X[:(n_int - N) * self.n_states , :], 2, 1)

        assert self.x0.shape[0] == self.U.shape[0] == self.target.shape[0] == self.num_samples, (self.x0.shape[0], self.U.shape[0], self.target.shape[0], self.num_samples)
        assert self.x0.shape[1] == self.n_states, (self.x0.shape[1], self.n_states)
        assert self.U.shape[1] == self.n_inputs * N, (self.U.shape[1], self.n_inputs * N)
        assert self.target.shape[1] == self.n_states * N, (self.target.shape[1], self.n_states * N)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x0[idx], self.U[idx], self.target[idx]


def sliding_window_2d(tensor, m, N):
    n1, n2 = tensor.shape
    assert n1 % m == 0
    # Number of rows in the output
    out_rows = (n1 // m + 1 - N) * n2
    result = torch.empty((out_rows, N * m), dtype=tensor.dtype)

    # Fill the result tensor
    idx = 0
    for col in range(n2):
        for row in range(n1 // m - N + 1):
            result[idx] = tensor[row:row + N * m, col]
            idx += 1

    return result