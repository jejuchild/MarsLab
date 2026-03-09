"""
PINN for Mars thermal inversion — V3.
Fixes: exp(raw) k-output, separate optimizers, balanced physics weight.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

RHO = 1500.0; C_P = 627.9; RHO_CP = RHO * C_P
MARS_SOL = 88_775.0; MARS_YEAR_SOLS = 668.6
MARS_YEAR_SEC = MARS_YEAR_SOLS * MARS_SOL

@dataclass
class PINNConfig:
    n_hidden_T: int = 4; n_neurons_T: int = 64
    n_hidden_k: int = 3; n_neurons_k: int = 32
    n_harmonics_diurnal: int = 4; n_harmonics_seasonal: int = 3
    n_depth_modes: int = 4
    k_min: float = 0.005; k_max: float = 4.0
    T_min: float = 140.0; T_max: float = 300.0
    z_max: float = 3.0
    eps_z: float = 0.002; eps_d: float = 0.01; eps_s: float = 0.005
    w_physics: float = 1.0; w_data: float = 10.0
    w_k_smooth: float = 1e-5; w_k_bounds: float = 0.01
    lr_T: float = 1e-3; lr_k: float = 1e-2
    n_epochs: int = 5000; batch_colloc: int = 4096
    scheduler_step: int = 2000; scheduler_gamma: float = 0.5
    k_only_interval: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class TemperatureNet(nn.Module):
    def __init__(self, nh=4, nn_=64, nhd=4, nhs=3, ndz=4):
        super().__init__()
        self.nhd, self.nhs, self.ndz = nhd, nhs, ndz
        dim = 1 + 2*nhd + 2*nhs + 2*ndz
        layers = [nn.Linear(dim, nn_), nn.Tanh()]
        for _ in range(nh - 1):
            layers += [nn.Linear(nn_, nn_), nn.Tanh()]
        layers.append(nn.Linear(nn_, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, z, td, ts):
        z, td, ts = z.view(-1,1), td.view(-1,1), ts.view(-1,1)
        f = [z]
        for n in range(1, self.nhd+1):
            p = 2*np.pi*n*td; f += [torch.sin(p), torch.cos(p)]
        for n in range(1, self.nhs+1):
            p = 2*np.pi*n*ts; f += [torch.sin(p), torch.cos(p)]
        for n in range(1, self.ndz+1):
            p = np.pi*n*z; f += [torch.sin(p), torch.cos(p)]
        return self.net(torch.cat(f, 1))

class ConductivityNet(nn.Module):
    """k = exp(raw). No sigmoid saturation. dk/d(raw) = k."""
    def __init__(self, nh=3, nn_=32, k_min=0.005, k_max=4.0):
        super().__init__()
        self.log_k_min = float(np.log(k_min))
        self.log_k_max = float(np.log(k_max))
        layers = [nn.Linear(1, nn_), nn.Tanh()]
        for _ in range(nh - 1):
            layers += [nn.Linear(nn_, nn_), nn.Tanh()]
        layers.append(nn.Linear(nn_, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5); nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.net[-1].bias.fill_(0.5*(self.log_k_min + self.log_k_max))

    def forward(self, z):
        return torch.exp(self.net(z.view(-1,1)).clamp(-8., 3.))

class ThermalPINN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.device = torch.device(cfg.device)
        self.T_net = TemperatureNet(cfg.n_hidden_T, cfg.n_neurons_T,
            cfg.n_harmonics_diurnal, cfg.n_harmonics_seasonal, cfg.n_depth_modes)
        self.k_net = ConductivityNet(cfg.n_hidden_k, cfg.n_neurons_k,
            cfg.k_min, cfg.k_max)
        self.to(self.device)

    def predict_T(self, z, td, ts):
        r = self.T_net(z, td, ts)
        return self.config.T_min + (self.config.T_max-self.config.T_min)*0.5*(r+1)

    def predict_k(self, z):
        return self.k_net(z)

    def physics_loss_fd(self, zc, tdc, tsc):
        ez, ed, es = self.config.eps_z, self.config.eps_d, self.config.eps_s
        zc = zc.clamp(2*ez, 1-2*ez)
        Tdp = self.predict_T(zc, tdc+ed, tsc)
        Tdm = self.predict_T(zc, tdc-ed, tsc)
        dTtd = (Tdp-Tdm)/(2*ed)
        Tsp = self.predict_T(zc, tdc, tsc+es)
        Tsm = self.predict_T(zc, tdc, tsc-es)
        dTts = (Tsp-Tsm)/(2*es)
        dTdt = dTtd/MARS_SOL + dTts/MARS_YEAR_SEC

        Tc = self.predict_T(zc, tdc, tsc)
        Tzp = self.predict_T(zc+ez, tdc, tsc)
        Tzm = self.predict_T(zc-ez, tdc, tsc)
        kp = self.predict_k(zc+.5*ez)
        km = self.predict_k(zc-.5*ez)
        fp = kp*(Tzp-Tc)/ez; fm = km*(Tc-Tzm)/ez
        div = (fp-fm)/ez/(self.config.z_max**2)

        res = RHO_CP*dTdt - div
        cs = RHO_CP*0.5*(self.config.T_max-self.config.T_min)*2*np.pi/MARS_SOL
        return ((res/max(cs,1e-10))**2).mean()

    def data_loss(self, zo, tdo, tso, To):
        Tp = self.predict_T(zo, tdo, tso)
        s = self.config.T_max - self.config.T_min
        return ((Tp - To.view(-1,1))**2).mean() / (s**2)

    def k_smooth_loss(self, zp):
        e = self.config.eps_z; z = zp.clamp(e,1-e)
        kc = self.predict_k(z); kp = self.predict_k(z+e); km = self.predict_k(z-e)
        d2k = (kp-2*kc+km)/(e**2)
        return ((d2k/(self.config.k_max-self.config.k_min))**2).mean()

def train_pinn(model, data, cfg):
    dev = model.device; zm = float(data["z"].max()); cfg.z_max = zm
    zo = torch.tensor(data.get("z_obs", np.zeros_like(data["t_obs_diurnal"]))/zm,
                      dtype=torch.float32).to(dev)
    tdo = torch.tensor(data["t_obs_diurnal"], dtype=torch.float32).to(dev)
    tso = torch.tensor(data["t_obs_seasonal"], dtype=torch.float32).to(dev)
    To = torch.tensor(data["T_obs"], dtype=torch.float32).to(dev)
    zca = torch.tensor(data["z_colloc"]/zm, dtype=torch.float32).to(dev)
    tdca = torch.tensor(data["t_colloc_diurnal"], dtype=torch.float32).to(dev)
    tsca = torch.tensor(data["t_colloc_seasonal"], dtype=torch.float32).to(dev)

    optT = torch.optim.Adam(model.T_net.parameters(), lr=cfg.lr_T)
    optk = torch.optim.Adam(model.k_net.parameters(), lr=cfg.lr_k)
    schT = torch.optim.lr_scheduler.StepLR(optT, cfg.scheduler_step, cfg.scheduler_gamma)
    schk = torch.optim.lr_scheduler.StepLR(optk, cfg.scheduler_step, cfg.scheduler_gamma)

    hist = {"loss_total":[],"loss_physics":[],"loss_data":[],"loss_smooth":[]}
    rng = np.random.default_rng(0)
    ne = cfg.n_epochs; p1=int(.20*ne); p2=int(.60*ne)
    lkmin, lkmax = float(np.log(cfg.k_min)), float(np.log(cfg.k_max))

    for ep in range(ne):
        model.train(); optT.zero_grad(); optk.zero_grad()
        if ep < p1:
            wp, wd = cfg.w_physics*0.5, cfg.w_data*3.0
        elif ep < p2:
            pr = (ep-p1)/max(p2-p1,1)
            wp = cfg.w_physics*(0.5+9.5*pr); wd = cfg.w_data*(3.0-1.0*pr)
        else:
            wp, wd = cfg.w_physics*10.0, cfg.w_data*2.0

        idx = rng.choice(len(zca), size=min(cfg.batch_colloc,len(zca)), replace=False)
        zc, tdc, tsc = zca[idx], tdca[idx], tsca[idx]

        lp = model.physics_loss_fd(zc, tdc, tsc)
        ld = model.data_loss(zo, tdo, tso, To)
        zs = torch.linspace(.01,.99,50,device=dev)
        ls = model.k_smooth_loss(zs)
        ks = model.predict_k(zs); lk = torch.log(ks+1e-10)
        lb = (torch.relu(lkmin-lk)**2 + torch.relu(lk-lkmax)**2).mean()
        loss = wp*lp + wd*ld + cfg.w_k_smooth*ls + cfg.w_k_bounds*lb

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optT.step(); optk.step(); schT.step(); schk.step()

        if ep % cfg.k_only_interval == 0 and ep >= p1:
            for p in model.T_net.parameters(): p.requires_grad_(False)
            optk.zero_grad()
            lko = model.physics_loss_fd(zc, tdc, tsc)
            lko.backward()
            torch.nn.utils.clip_grad_norm_(model.k_net.parameters(), 1.0)
            optk.step()
            for p in model.T_net.parameters(): p.requires_grad_(True)

        hist["loss_total"].append(loss.item())
        hist["loss_physics"].append(lp.item())
        hist["loss_data"].append(ld.item())
        hist["loss_smooth"].append(ls.item())

        if (ep+1) % 500 == 0 or ep == 0:
            ph = "P1" if ep<p1 else ("P2" if ep<p2 else "P3")
            gk = sum(p.grad.norm().item() for p in model.k_net.parameters()
                     if p.grad is not None)
            gt = sum(p.grad.norm().item() for p in model.T_net.parameters()
                     if p.grad is not None)
            logger.info("E%5d [%s] L=%.2e Ph=%.2e Da=%.2e wp=%.1f wd=%.1f gT=%.1e gk=%.1e",
                        ep+1, ph, loss.item(), lp.item(), ld.item(), wp, wd, gt, gk)
            model.eval()
            with torch.no_grad():
                zt = torch.tensor([0.,.05,.1,.167,.333,.5,.667,1.], device=dev)
                kt = model.predict_k(zt).cpu().numpy().flatten()
                d = zt.cpu().numpy()*cfg.z_max
                logger.info("  k: %s", "  ".join(f"{dd:.1f}m={kk:.4f}" for dd,kk in zip(d,kt)))
    return hist

def evaluate_pinn(model, data):
    dev = model.device; cfg = model.config
    z = data["z"]; zn = torch.tensor(z/cfg.z_max, dtype=torch.float32).to(dev)
    model.eval()
    with torch.no_grad():
        kp = model.predict_k(zn).cpu().numpy().flatten()
    kt = data["k_true"]
    td = torch.tensor(data["t_obs_diurnal"], dtype=torch.float32).to(dev)
    ts = torch.tensor(data["t_obs_seasonal"], dtype=torch.float32).to(dev)
    # Only evaluate on surface observations
    mask = data.get("z_obs", np.zeros(len(data["t_obs_diurnal"]))) < 0.001
    td_s, ts_s = td[mask], ts[mask]
    z0 = torch.zeros_like(td_s)
    with torch.no_grad():
        Tp = model.predict_T(z0, td_s, ts_s).cpu().numpy().flatten()
    Tt = data["T_obs"][mask]

    kr = float(np.sqrt(np.mean((kp-kt)**2)))
    Tr = float(np.sqrt(np.mean((Tp-Tt)**2)))
    TIp = np.sqrt(np.clip(kp,0,None)*RHO*C_P)
    TIt = np.sqrt(kt*RHO*C_P)
    bi = np.argmax(np.abs(np.gradient(kp))); bd = z[bi]
    ka = np.mean(kp[:max(bi,1)]); kb = np.mean(kp[min(bi+10,len(kp)):])

    logger.info("="*50)
    logger.info("k_RMSE=%.4f  T_RMSE=%.2f K  boundary=%.2fm", kr, Tr, bd)
    logger.info("TI: pred %.0f-%.0f  true %.0f-%.0f  k_above=%.4f k_below=%.4f",
                TIp.min(), TIp.max(), TIt.min(), TIt.max(), ka, kb)
    logger.info("="*50)
    return {"z":z, "k_pred":kp, "k_true":kt, "TI_pred":TIp, "TI_true":TIt,
            "T_pred_surface":Tp, "T_true_surface":Tt,
            "k_rmse":kr, "T_rmse":Tr, "boundary_depth":bd}
