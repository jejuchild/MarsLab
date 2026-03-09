"""Synthetic data generation for PINN — V3 with subsurface obs support."""
from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)
MARS_SOL = 88_775.0; MARS_YEAR_SOLS = 668.6
MARS_YEAR_SEC = MARS_YEAR_SOLS * MARS_SOL
RHO = 1500.0; C_P = 627.9

@dataclass
class SyntheticConfig:
    z_max: float = 3.0; n_z: int = 200; n_sols: int = 668
    dt_per_sol: int = 144; spinup_sols: int = 668
    T_mean: float = 210.0; T_amp_diurnal: float = 40.0
    T_amp_seasonal: float = 20.0; latitude: float = 45.0
    rho: float = RHO; c_p: float = C_P

def two_layer_k_profile(z, k_upper=0.02, k_lower=2.0,
                         boundary_depth=0.5, transition_width=0.05):
    return k_upper + (k_lower-k_upper)/(1+np.exp(-(z-boundary_depth)/transition_width))

def three_layer_k_profile(z, k1=0.02, k2=0.1, k3=2.0,
                           d1=0.2, d2=0.8, tw=0.05):
    s1 = 1/(1+np.exp(-(z-d1)/tw)); s2 = 1/(1+np.exp(-(z-d2)/tw))
    return k1 + (k2-k1)*s1 + (k3-k2)*s2

def surface_forcing(t, cfg):
    wd = 2*np.pi/MARS_SOL; ws = 2*np.pi/MARS_YEAR_SEC
    return cfg.T_mean + cfg.T_amp_diurnal*np.sin(wd*t) + cfg.T_amp_seasonal*np.sin(ws*t)

def solve_heat_equation(k_z, cfg):
    nz = cfg.n_z; dz = cfg.z_max/(nz-1); z = np.linspace(0,cfg.z_max,nz)
    nt_total = (cfg.n_sols+cfg.spinup_sols)*cfg.dt_per_sol
    dt = MARS_SOL/cfg.dt_per_sol; t_all = np.arange(nt_total)*dt
    rc = cfg.rho*cfg.c_p
    kh = 2*k_z[:-1]*k_z[1:]/(k_z[:-1]+k_z[1:])
    r = kh*dt/(rc*dz*dz)

    a, b, c = np.zeros(nz), np.zeros(nz), np.zeros(nz)
    b[0] = 1.0
    a[1:-1] = -.5*r[:-1]; b[1:-1] = 1+.5*(r[:-1]+r[1:]); c[1:-1] = -.5*r[1:]
    a[-1] = -.5*r[-1]; b[-1] = 1+.5*r[-1]

    cp, w = np.zeros(nz), np.zeros(nz)
    w[0] = b[0]; cp[0] = c[0]/w[0]
    for i in range(1,nz):
        w[i] = b[i]-a[i]*cp[i-1]
        if abs(w[i]) < 1e-30: w[i] = 1e-30
        cp[i] = c[i]/w[i]

    Ts = surface_forcing(t_all, cfg)
    T = np.full(nz, float(cfg.T_mean))
    nsp = cfg.spinup_sols; nt_out = cfg.n_sols*cfg.dt_per_sol
    Tout = np.zeros((nt_out, nz))
    rhs, dp, x = np.zeros(nz), np.zeros(nz), np.zeros(nz)

    for n in range(nt_total):
        rhs[0] = Ts[n]
        rhs[1:-1] = T[1:-1] + .5*r[:-1]*(T[:-2]-T[1:-1]) + .5*r[1:]*(T[2:]-T[1:-1])
        rhs[-1] = T[-1] + .5*r[-1]*(T[-2]-T[-1])
        dp[0] = rhs[0]/w[0]
        for i in range(1,nz): dp[i] = (rhs[i]-a[i]*dp[i-1])/w[i]
        x[-1] = dp[-1]
        for i in range(nz-2,-1,-1): x[i] = dp[i]-cp[i]*x[i+1]
        T[:] = x
        oi = n - nsp*cfg.dt_per_sol
        if oi >= 0: Tout[oi] = T

    return Tout, z, np.arange(nt_out)*dt

def generate_synthetic_dataset(k_profile_fn=None, config=None,
                                n_surface_obs=3000, n_colloc=8000,
                                subsurface_depths=None,
                                subsurface_obs_per_depth=200, seed=42):
    if config is None: config = SyntheticConfig()
    z = np.linspace(0, config.z_max, config.n_z)
    if k_profile_fn is None: k_profile_fn = two_layer_k_profile
    k_true = k_profile_fn(z)

    logger.info("FDM solve (nz=%d, sols=%d+%d spinup)...", config.n_z, config.n_sols, config.spinup_sols)
    Tf, z, t = solve_heat_equation(k_true, config)
    logger.info("T range: %.1f-%.1f K, shape=%s", Tf.min(), Tf.max(), Tf.shape)

    rng = np.random.default_rng(seed); nt = len(t)
    oi = rng.choice(nt, size=min(n_surface_obs,nt), replace=False); oi.sort()
    t_obs, T_obs, z_obs = t[oi], Tf[oi,0], np.zeros(len(oi))

    if subsurface_depths:
        for d in subsurface_depths:
            zi = np.argmin(np.abs(z-d))
            si = rng.choice(nt, size=min(subsurface_obs_per_depth,nt), replace=False); si.sort()
            t_obs = np.concatenate([t_obs, t[si]])
            T_obs = np.concatenate([T_obs, Tf[si,zi]])
            z_obs = np.concatenate([z_obs, np.full(len(si), z[zi])])
            logger.info("  Sub z=%.2fm: %d pts T=%.1f-%.1f K", z[zi], len(si), Tf[si,zi].min(), Tf[si,zi].max())

    td = (t_obs%MARS_SOL)/MARS_SOL; ts = t_obs/MARS_YEAR_SEC

    nh = n_colloc//2
    zc = np.concatenate([rng.beta(2,5,nh)*config.z_max,
                         rng.uniform(0,config.z_max,n_colloc-nh)]).astype(np.float32)
    tdc = rng.uniform(0,1,n_colloc).astype(np.float32)
    ny = config.n_sols/MARS_YEAR_SOLS
    tsc = rng.uniform(0,ny,n_colloc).astype(np.float32)

    logger.info("TI: upper=%.0f lower=%.0f | Obs: %d total",
                np.sqrt(k_true[0]*config.rho*config.c_p),
                np.sqrt(k_true[-1]*config.rho*config.c_p), len(T_obs))

    return {"z":z.astype(np.float32), "t":t.astype(np.float32),
            "T_full":Tf.astype(np.float32), "k_true":k_true.astype(np.float32),
            "t_obs":t_obs.astype(np.float32), "T_obs":T_obs.astype(np.float32),
            "z_obs":z_obs.astype(np.float32),
            "t_obs_diurnal":td.astype(np.float32), "t_obs_seasonal":ts.astype(np.float32),
            "z_colloc":zc, "t_colloc_diurnal":tdc, "t_colloc_seasonal":tsc,
            "config":config}
