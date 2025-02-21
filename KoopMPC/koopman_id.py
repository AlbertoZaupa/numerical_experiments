import sys
sys.path.append('../src')
import pykoopman as pk
from pykoopman.common.examples import vdp_osc, rk4
import numpy as np
import numpy.random as rnd
np.random.seed(42)  # for reproducibility

import warnings
warnings.filterwarnings('ignore')

def fit_Koopman(dynamics=vdp_osc, n_states=2, n_inputs=1, dT=0.01, n_traj=200, n_int=1000):
    t = np.arange(0, n_int*dT, dT)

    # Uniform random distributed forcing in [-1, 1]
    u = 2*rnd.random([n_int, n_traj])-1

    # Uniform distribution of initial conditions
    x = 2*rnd.random([n_states, n_traj])-1

    # Init
    X = np.zeros((n_states, n_int*n_traj))
    Y = np.zeros((n_states, n_int*n_traj))
    U = np.zeros((n_inputs, n_int*n_traj))

    # Integrate
    for step in range(n_int):
        y = rk4(0, x, u[step, :], dT, dynamics)
        X[:, (step)*n_traj:(step+1)*n_traj] = x
        Y[:, (step)*n_traj:(step+1)*n_traj] = y
        U[:, (step)*n_traj:(step+1)*n_traj] = u[step, :]
        x = y
    EDMDc = pk.regression.EDMDc()
    centers = np.random.uniform(-1,1,(2,5))
    RBF = pk.observables.RadialBasisFunction(
        rbf_type="thinplate",
        n_centers=centers.shape[1],
        centers=centers,
        kernel_width=1,
        polyharmonic_coeff=1,
        include_state=True,
    )

    model = pk.Koopman(observables=RBF, regressor=EDMDc)
    model.fit(X.T, y=Y.T, u=U.T)
    return model.A, model.B, model.C, model.phi
