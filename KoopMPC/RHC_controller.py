from solver.qpsolver import QPsolver
import numpy as np
import matplotlib.pyplot as plt

TOL = 1e-4

class RHC_controller():
    
    def __init__(self, A, C, B, Q, R, Pf, N, Ex, ex, Eu, eu, Ef, ef):
        self.H, self.L, self.E, self.F, self.e = self.condense(A, C, B, Q, R, Pf, N, Ex, Eu, Ef, ex, eu, ef)
        self.solver = QPsolver()
        self.solver.setup(P=self.H, q=np.zeros(self.L.shape[0]), G=self.E,
                          l=np.full((self.E.shape[0],), fill_value=-np.inf),
                          u=self.e, eps_abs=TOL)

    def condense(self, A, B, C, Q, R, Pf, N, Ey, Eu, Ef, ex, eu, ef):
        # Dimensions
        self.nx = A.shape[1]
        self.nu = B.shape[1]
        self.ny = C.shape[0]
        self.N = N
        self.nu_c = eu.shape[0]

        # Construct block diagonal matrices
        Q_bar = np.kron(np.eye(N), Q)
        Q_bar[-self.ny:, -self.ny:] = Pf
        R_bar = np.kron(np.eye(N), R)

        # Compute 'Impulse Response'
        G = np.zeros((self.ny * N, self.nu * N))
        A_kB = B
        for k in range(N):
            for i in range(k, N):
                G[i * self.ny:(i + 1) * self.ny, (i-k) * self.nu:(i-k + 1) * self.nu] = C @ A_kB
            A_kB = A @ A_kB

        # Compute Hessian
        H = R_bar + G.T @ Q_bar @ G

        # Compute L
        M = np.zeros((self.ny * N, self.nx))
        Ak = A
        for k in range(N):
            M[k * self.ny:(k + 1) * self.ny, :] =  C @ Ak
            Ak = A @ Ak
        L = G.T @ Q_bar @ M

        # Compute E, e
        E1 = np.kron(np.eye(N), Eu)
        e1 = np.zeros((eu.shape[0] * N, 1))
        for k in range(N):
            e1[k * self.nu_c:(k + 1) * self.nu_c, :] = eu

        E_ = np.kron(np.eye(N), Ey)
        E_[-Ey.shape[0]:, -self.ny:] = Ef
        E2 = E_ @ G
        E = np.vstack((E1, E2))
        e2 = np.zeros((ex.shape[0] * N, 1))
        for k in range(N):
            e2[k * ex.shape[0]:(k + 1) * ex.shape[0], :] = ex
        e2[-ex.shape[0]:, :] = ef
        e = np.vstack((e1, e2))

        # Compute F
        F = E_ @ M

        return H, L, E, F, e

    def solve(self, x0):
        g = self.L @ x0
        g = g.reshape(-1)
        upp = self.e - np.vstack((np.zeros((self.N * self.nu_c, 1)), -self.F @ x0))
        upp = upp.reshape(-1)
        self.solver.update(q=g, u=upp)
        results = self.solver.solve()
        #assert results.info.status == 'solved', (results.info.pri_res, results.info.dua_res)
        return results.x[:self.nu]


if __name__ == "__main__":
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])
    C = np.eye(2)
    Q = np.eye(2)
    R = 1
    Pf = Q
    N = 200
    M = 10
    Ex = np.vstack((np.eye(2), -np.eye(2)))
    Ef = Ex
    Eu = np.vstack((np.eye(1), -np.eye(1)))
    ex = np.array([[35], [30], [35], [30]])
    ef = ex
    eu = np.array([[1], [1]])

    x0 = np.array([[10], [5]])
    solver = RHC_controller(A, B, C, Q, R, Pf, N, Ex, ex, Eu, eu, Ef, ef)
    x1 = np.zeros(M)
    x2 = np.zeros(M)
    u = np.zeros(M)
    for t in range(M):
        x1[t:t+1] = x0[0:1,:]
        x2[t:t+1] = x0[1:2,:]
        u_t = solver.solve(x0)
        u[t] = u_t
        x0 = A @ x0 + (B @ u_t).reshape((2,1))

    plt.figure(figsize=(8, 6))
    plt.plot(x1, x2, marker='o', markersize=3, linestyle='-', color='blue')
    plt.xlabel('x1(t)')
    plt.ylabel('x2(t)')
    plt.title('State trajectory')
    plt.grid(True)
    plt.axis('equal')  # Ensure the scale is equal on both axes
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(range(M),u, marker='o', markersize=3, linestyle='-', color='blue')
    print(u)
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.title('Input signal')
    plt.grid(True)
    plt.axis('equal')  # Ensure the scale is equal on both axes
    plt.show()

