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
                        check_interval=25):
        
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

