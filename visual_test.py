import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import make_loaders
from utils import FrozenFullRecon


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
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(title)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def compute_val_metrics(S_true: np.ndarray, S_recon: np.ndarray, eps: float = 1e-8):
    """
    Calcola RMSE, MRAE, PSNR e SAM sull'intero validation set.

    S_true  : array N×L
    S_recon : array N×L
    """
    assert S_true.shape == S_recon.shape, f"Shape mismatch: {S_true.shape} vs {S_recon.shape}"

    # -------------------------
    # MSE per spettro (su tutte le bande)
    # -------------------------
    mse_per_sample = np.mean((S_true - S_recon) ** 2, axis=1)

    # RMSE globale
    rmse = np.sqrt(np.mean(mse_per_sample))

    # MRAE globale
    mrae = np.mean(np.abs(S_true - S_recon) / (S_true + eps))

    # PSNR globale (MAX=1)
    psnr = 20 * np.log10(1.0 / np.sqrt(np.mean(mse_per_sample) + eps))

    # -------------------------
    # SAM (Spectral Angle Mapper)
    # -------------------------
    dot = np.sum(S_true * S_recon, axis=1)
    norm_gt = np.linalg.norm(S_true, axis=1)
    norm_recon = np.linalg.norm(S_recon, axis=1)

    cos_theta = dot / (norm_gt * norm_recon + eps)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # per stabilità numerica

    sam = np.mean(np.arccos(cos_theta))  # SAM medio (in radianti)

    return rmse, mrae, psnr, sam



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

    Assunzione: il loader restituisce (x, y),
    dove:
      - x è lo spettro o il tensore da cui ricavare lo spettro
      - y è la label (classe)
    """
    recon_model.eval()
    recon_model.to(device)

    all_mse = []
    all_y = []
    all_s_true = []
    all_s_recon = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)  # (B,121) oppure (B,1,121) oppure (B,121,H,W)
            y = y.to(device)  # (B,)

            # porta tutto a spettro [B, L]
            if x.dim() == 4:
                # (B, C, H, W) -> media spaziale sulle dimensioni H,W
                # ipotesi: C = 121
                s_true = x.mean(dim=(2, 3))
            else:
                # (B, L) oppure (B, 1, L)
                s_true = x
                if s_true.dim() == 3 and s_true.shape[1] == 1:
                    s_true = s_true[:, 0, :]

            # ricostruzione: il modello prende in input s_true (adatta se il tuo modello si aspetta altro)
            s_recon = recon_model(s_true)  # (B, L) atteso

            # MSE spettrale per campione
            mse = ((s_recon - s_true) ** 2).mean(dim=1)  # (B,)

            all_mse.append(mse.cpu())
            all_y.append(y.cpu())
            all_s_true.append(s_true.cpu())
            all_s_recon.append(s_recon.cpu())

    all_mse = torch.cat(all_mse).numpy()           # (N,)
    all_y = torch.cat(all_y).numpy().astype(int)   # (N,)
    all_s_true = torch.cat(all_s_true).numpy()     # (N,L)
    all_s_recon = torch.cat(all_s_recon).numpy()   # (N,L)

    return all_mse, all_y, all_s_true, all_s_recon


# ---------- selezione best/worst e plot ----------
def plot_best_worst_per_class(
    mse: np.ndarray,
    y: np.ndarray,
    s_true: np.ndarray,
    s_recon: np.ndarray,
    out_dir: str = "debug_plots_recon",
    num_best: int = 5,
    num_worst: int = 5,
    wavelengths: np.ndarray = None
):
    os.makedirs(out_dir, exist_ok=True)

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
            title = f"Class {cls} - BEST #{rank + 1} - MSE={mse[i]:.4f}"
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
            title = f"Class {cls} - WORST #{rank + 1} - MSE={mse[i]:.4f}"
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

    # ====== RACCOLTA STATISTICHE SUL VALIDATION ======
    mse, y, s_true, s_recon = collect_recon_stats(
        recon_model=recon_model,
        loader=val_loader,
        device=device,
    )

    rmse, mrae, psnr, sam = compute_val_metrics(s_true, s_recon)
    print("===== Validation Metrics =====")
    print(f"Validation RMSE: {rmse:.6f}")
    print(f"Validation MRAE: {mrae:.6f}")
    print(f"Validation PSNR: {psnr:.3f} dB")
    print(f"Validation SAM : {sam:.6f} rad  ({sam * 180 / np.pi:.3f} deg)")
    print("================================\n")

    # ====== PLOT BEST/WORST PER CLASSE ======
    out_dir = os.path.join(save_dir, "recon_debug_plots")
    plot_best_worst_per_class(
        mse=mse,
        y=y,
        s_true=s_true,
        s_recon=s_recon,
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


