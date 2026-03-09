#!/usr/bin/env python3
"""Phase 1: PINN V3 synthetic validation."""
from __future__ import annotations
import logging, time
from pathlib import Path
import numpy as np
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
logger = logging.getLogger(__name__)

def visualize(res, data, hist, out):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError: return
    fig, ax = plt.subplots(2,2,figsize=(14,10))
    fig.suptitle("PINN V3 — k(z) Recovery", fontsize=14)
    z = res["z"]
    ax[0,0].plot(z, res["k_true"], "b-", lw=2, label="True")
    ax[0,0].plot(z, res["k_pred"], "r--", lw=2, label="Pred")
    ax[0,0].set(xlabel="Depth [m]", ylabel="k [W/m/K]", xlim=(0,2))
    ax[0,0].set_title(f"k(z) RMSE={res['k_rmse']:.4f}"); ax[0,0].legend(); ax[0,0].grid(True,alpha=.3)
    ts = data["t_obs"]/88775.; mask = data.get("z_obs",np.zeros(1))<.001
    if mask.sum()>10:
        m3 = ts[mask]<3
        if m3.sum()>5:
            ax[0,1].plot(ts[mask][m3], res["T_true_surface"][m3], "b.", ms=2, alpha=.6)
            ax[0,1].plot(ts[mask][m3], res["T_pred_surface"][m3], "r.", ms=2, alpha=.6)
    ax[0,1].set(xlabel="Sols", ylabel="T [K]"); ax[0,1].set_title(f"Diurnal RMSE={res['T_rmse']:.2f}K")
    ax[0,1].grid(True,alpha=.3)
    ax[1,0].plot(ts[mask], res["T_true_surface"], "b.", ms=1, alpha=.3)
    ax[1,0].plot(ts[mask], res["T_pred_surface"], "r.", ms=1, alpha=.3)
    ax[1,0].set(xlabel="Sols", ylabel="T [K]"); ax[1,0].set_title("Full Year"); ax[1,0].grid(True,alpha=.3)
    ep = np.arange(1,len(hist["loss_total"])+1)
    ax[1,1].semilogy(ep, hist["loss_data"], "b-", alpha=.7, label="Data")
    ax[1,1].semilogy(ep, hist["loss_physics"], "r-", alpha=.7, label="Physics")
    n=len(ep); ax[1,1].axvline(int(.2*n),color="gray",ls=":"); ax[1,1].axvline(int(.6*n),color="gray",ls="--")
    ax[1,1].set(xlabel="Epoch",ylabel="Loss"); ax[1,1].legend(fontsize=8); ax[1,1].grid(True,alpha=.3)
    plt.tight_layout(); fig.savefig(out/"pinn_v3.png", dpi=150); plt.close(fig)
    logger.info("Plot saved: %s", out/"pinn_v3.png")

def main():
    from .synthetic import SyntheticConfig, generate_synthetic_dataset, two_layer_k_profile
    from .pinn_model import PINNConfig, ThermalPINN, train_pinn, evaluate_pinn
    import torch

    sc = SyntheticConfig(z_max=3, n_z=200, n_sols=668, dt_per_sol=144, spinup_sols=668,
                         T_mean=210, T_amp_diurnal=40, T_amp_seasonal=20)
    pc = PINNConfig(n_hidden_T=4, n_neurons_T=64, n_hidden_k=3, n_neurons_k=32,
                    k_min=.005, k_max=4., T_min=140, T_max=300,
                    w_physics=1., w_data=10., w_k_smooth=1e-5, w_k_bounds=.01,
                    lr_T=1e-3, lr_k=1e-2, n_epochs=5000, batch_colloc=4096,
                    scheduler_step=2000, scheduler_gamma=.5, k_only_interval=5)

    logger.info("="*60)
    logger.info("PINN V3: exp(k) + separate optimizers + subsurface obs")
    logger.info("="*60)

    t0 = time.time()
    data = generate_synthetic_dataset(two_layer_k_profile, sc, 3000, 8000,
                                       subsurface_depths=[.1,.3,.5,1.], subsurface_obs_per_depth=200)
    tg = time.time()-t0; logger.info("FDM: %.1fs", tg)

    model = ThermalPINN(pc)
    logger.info("Params: %d  Device: %s", sum(p.numel() for p in model.parameters()), model.device)
    model.eval()
    with torch.no_grad():
        zt = torch.tensor([0.,.167,.5,1.])
        ki = model.predict_k(zt).numpy().flatten()
        logger.info("Init k: %s", "  ".join(f"{d*3:.1f}m={k:.4f}" for d,k in zip(zt.numpy(),ki)))

    t0 = time.time()
    hist = train_pinn(model, data, pc)
    tt = time.time()-t0; logger.info("Train: %.1fs (%.1fms/ep)", tt, 1000*tt/pc.n_epochs)

    res = evaluate_pinn(model, data)
    od = Path(__file__).resolve().parents[2]/"data"/"thermal_pinn"; od.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), od/"pinn_v3_weights.pt")
    np.savez_compressed(od/"pinn_v3_results.npz", z=res["z"], k_pred=res["k_pred"],
        k_true=res["k_true"], TI_pred=res["TI_pred"], TI_true=res["TI_true"],
        loss_total=np.array(hist["loss_total"]), loss_physics=np.array(hist["loss_physics"]),
        loss_data=np.array(hist["loss_data"]))
    visualize(res, data, hist, od)
    logger.info("DONE: k_RMSE=%.4f T_RMSE=%.2f boundary=%.2fm [%.0fs total]",
                res["k_rmse"], res["T_rmse"], res["boundary_depth"], tg+tt)
    return res

if __name__ == "__main__": main()
