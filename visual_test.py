import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import make_loaders
from utils import FrozenFullRecon
# importa qui la tua FrozenFullRecon (ad es. dallo stesso file o dal modulo giusto)
# from models.recon import FrozenFullRecon
# oppure se è nello stesso file, ignora l'import

# ---------------- util ----------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- plotting helper ----------
def plot_spectrum_pair(
    s_true: np.ndarray,
    s_recon: np.ndarray,
    title: str,
    out_path: str,
    wavelengths: np.ndarray = None
):
    """Plot di uno spettro GT vs ricostruito e salvataggio su file."""
    if wavelengths is None:
        wavelengths = np.arange(len(s_true))

    plt.figure()
    plt.plot(wavelengths, s_true, label="GT")
    plt.plot(wavelengths, s_recon, linestyle="--", label="Recon")
    plt.xlabel("Band index")
    plt.ylabel("Valore spettrale")
    plt.title(title)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------- raccolta statistiche ricostruzione ----------
def collect_recon_stats(
    recon_model: nn.Module,
    loader,
    device: torch.device,
):
    """
    Usa SOLO il modello di ricostruzione per:
      - ricostruire ogni spettro
      - calcolare MSE spettrale
      - salvare GT, Recon, label
    """
    recon_model.eval()
    recon_model.to(device)

    all_mse = []
    all_y = []
    all_s_true = []
    all_s_recon = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)  # (B,121) oppure (B,121,H,W)
            y = y.to(device)  # (B,)

            # porta tutto a spettro medio [B,121]
            if x.dim() == 4:
                # (B,121,H,W) -> media spaziale
                s_true = x.mean(dim=(2, 3))
                s_true = s_true.squeeze(1)
            else:
                s_true = x  # (B,121)
                s_true = s_true.squeeze(1)

            # ricostruzione
            s_recon = recon_model(s_true)  # (B,121)

            # MSE spettrale per campione
            mse = ((s_recon - s_true) ** 2).mean(dim=1)  # (B,)

            all_mse.append(mse.cpu())
            all_y.append(y.cpu())
            all_s_true.append(s_true.cpu())
            all_s_recon.append(s_recon.cpu())

    all_mse = torch.cat(all_mse).numpy()           # (N,)
    all_y = torch.cat(all_y).numpy().astype(int)   # (N,)
    all_s_true = torch.cat(all_s_true).numpy()     # (N,121)
    all_s_recon = torch.cat(all_s_recon).numpy()   # (N,121)

    return all_mse, all_y, all_s_true, all_s_recon


# ---------- selezione best/worst e plot ----------
def plot_best_worst_per_class(
    recon_model: nn.Module,
    val_loader,
    device: torch.device,
    out_dir: str = "debug_plots_recon",
    num_best: int = 5,
    num_worst: int = 5,
    wavelengths: np.ndarray = None
):
    os.makedirs(out_dir, exist_ok=True)

    mse, y, s_true, s_recon = collect_recon_stats(recon_model, val_loader, device)

    for cls in [0, 1]:
        idx_cls = np.where(y == cls)[0]
        if len(idx_cls) == 0:
            print(f"[WARN] Nessun campione per classe {cls}")
            continue

        # ordina per MSE crescente
        idx_sorted = idx_cls[np.argsort(mse[idx_cls])]
        best_idx = idx_sorted[:min(num_best, len(idx_sorted))]
        worst_idx = idx_sorted[-min(num_worst, len(idx_sorted)):]

        # BEST
        for rank, i in enumerate(best_idx):
            title = f"Class {cls} - BEST #{rank + 1} - MSE={np.mean(mse[i]):.4e}"
            out_path = os.path.join(out_dir, f"class{cls}_best_{rank+1}_idx{i}.png")
            plot_spectrum_pair(
                s_true[i],
                s_recon[i],
                title,
                out_path,
                wavelengths=wavelengths,
            )

        # WORST
        for rank, i in enumerate(worst_idx):
            title = f"Class {cls} - WORST #{rank + 1} - MSE={np.mean(mse[i]):.4}"
            out_path = os.path.join(out_dir, f"class{cls}_worst_{rank+1}_idx{i}.png")
            plot_spectrum_pair(
                s_true[i],
                s_recon[i],
                title,
                out_path,
                wavelengths=wavelengths,
            )

    print(f"Plot salvati in: {out_dir}")


# ---------------- main ----------------
def main(
        data_root: str,
        sensor_root: str,
        rgb: bool,
        ir: bool,
        save_dir: str = "runs/recon_eval",
        batch_size: int = 8,
        num_workers: int = 4,
        seed: int = 42,
        recon_ckpt: str = "",
        patch: bool = True
):
    set_seed(seed)

    # ci serve solo il val_loader, ma make_loaders restituisce anche il train
    train_loader, val_loader = make_loaders(
        data_root, sensor_root, rgb, ir,
        patch_mean=patch,
        batch_size=batch_size,
        num_workers=num_workers,
        val_ratio=0.2
    )
    del train_loader  # non lo usiamo

    # modello di ricostruzione (meas + decoder), SENZA classificazione
    recon_model = FrozenFullRecon(
        ckpt_path=recon_ckpt,
        spectral_sens_csv=sensor_root,
        recon_type=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # se vuoi usare le vere λ metti un array qui, altrimenti lascia None
    wavelengths = np.linspace(400, 1000, 121)
    # esempio:
    # wavelengths = np.linspace(400, 720, 121)

    out_dir = os.path.join(save_dir, "recon_debug_plots")
    plot_best_worst_per_class(
        recon_model,
        val_loader,
        device=device,
        out_dir=out_dir,
        num_best=5,
        num_worst=5,
        wavelengths=wavelengths,
    )


if __name__ == "__main__":

    arg = argparse.ArgumentParser()
    arg.add_argument("--data_root", type=str, required=True)
    arg.add_argument("--sensor_root", type=str, required=True)
    arg.add_argument("--save_dir", type=str, default="runs/recon_eval")
    arg.add_argument("--batch_size", type=int, default=8)
    arg.add_argument("--num_workers", type=int, default=4)
    arg.add_argument("--seed", type=int, default=42)
    arg.add_argument("--rgb", type=bool, default=False)
    arg.add_argument("--ir", type=bool, default=False)
    arg.add_argument("--patch_mean", type=bool, default=True)
    arg.add_argument("--recon_ckpt", type=str, required=True,
                     help="path ai pesi pre-addestrati di JointDualFilterMST (state_dict)")
    args = arg.parse_args()

    main(
        data_root=args.data_root,
        sensor_root=args.sensor_root,
        rgb=args.rgb,
        ir=args.ir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        recon_ckpt=args.recon_ckpt,
        patch=args.patch_mean,
    )

