import math, random
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

# === i tuoi moduli ===
from data_loader import FlourFolderDataset
from models.our import SPAN

# === import del ricostruttore (codice che mi hai passato) ===
# Assumo che le classi siano nello stesso progetto, altrimenti importa dal loro path
# from recon.mstpp import JointDualFilterMST
# Per comodità, copio qui la firma: JointDualFilterMST(k=3, sum_to_one=False)
from models.filter_opt import JointDualFilterMST  # <-- aggiorna il path giusto


# ---------------- util ----------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(root, sensor_root, rgb, ir, batch_size=8, num_workers=4, val_ratio=0.2, pin_memory=True):
    """
    Se esistono /train e /val sotto root li usa; altrimenti fa split random.
    """
    root = Path(root)
    full = FlourFolderDataset(root=root,
                              spectral_sens_csv=sensor_root,
                              rgb=rgb, ir=ir,
                              hsi_channels_first=False,  # True se i tuoi HSI sono (L,H,W)
                              illuminant_mode="planck",  # alogena
                              illuminant_T=2856.0)
    n = len(full)
    n_val = int(math.floor(n * val_ratio))
    n_train = n - n_val
    train_set, val_set = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    pw = num_workers > 0

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=pw
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=pw
    )
    return train_loader, val_loader


# ------------- LightningModule -------------
class LitReconThenSPAN(pl.LightningModule):
    """
    Pipeline: 4ch (RGB-IR) --[ JointDualFilterMST (frozen) ]--> HSI (121ch) --[ SPAN ]--> logit (B,1)
    Loss: BCEWithLogits
    """
    def __init__(self, se: bool, lr: float, recon_ckpt: str, recon_k: int = 3, recon_sum_to_one: bool = False):
        super().__init__()
        self.save_hyperparameters()

        # 1) Ricostruttore (congelato)
        self.recon = JointDualFilterMST()
        if recon_ckpt and len(recon_ckpt) > 0:
            ckpt = torch.load(recon_ckpt, map_location="cpu")
            # supporto sia {"state_dict": ...} (Lightning) sia state_dict diretto
            state = ckpt.get("state_dict", ckpt)
            # se è stato salvato con prefissi tipo "model." o simili, provo a ripulire
            cleaned = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("model.") or nk.startswith("recon."):
                    nk = nk.split(".", 1)[1]
                cleaned[nk] = v
            missing, unexpected = self.recon.load_state_dict(cleaned, strict=False)
            if len(missing) or len(unexpected):
                print(f"[WARN] Recon state_dict: missing={missing}, unexpected={unexpected}")
        # congelo
        for p in self.recon.parameters():
            p.requires_grad = False
        self.recon.eval()

        # 2) Classificatore SPAN: input = 121 canali
        in_channels = 121
        self.model = SPAN(num_in_ch=in_channels, feature_channels=48, bias=True, se=se)

        self.criterion = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def _reconstruct_hsi(self, x4):
        # x4: [B,4,H,W] -> [B,121,H,W]
        self.recon.eval()
        return self.recon(x4)

    def forward(self, x4):
        # pipeline completa
        with torch.no_grad():
            hsi = self._reconstruct_hsi(x4)
        logits = self.model(hsi)  # (B,1)
        return logits

    def _step(self, batch, stage: str):
        x4, y = batch                     # x4: [B,4,H,W], y: [B] o [B,1]
        with torch.no_grad():
            hsi = self._reconstruct_hsi(x4)
        logits = self.model(hsi)          # (B,1)
        y_int = y.long()
        y_f = y_int.float().unsqueeze(1)
        loss = self.criterion(logits, y_f)
        pred = (logits.view(-1) >= 0.5).long()
        acc = (pred == y_int.view(-1)).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.hparams.lr)  # ottimizziamo SOLO lo SPAN
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=50)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_fit_start(self):
        # assicura che il ricostruttore sia sullo stesso device
        self.recon.to(self.device)


# ---------------- main ----------------
def main(
        data_root: str,
        sensor_root: str,
        rgb: bool,
        ir: bool,
        save_dir: str = "runs/span_lightning",
        batch_size: int = 8,
        num_workers: int = 4,
        lr: float = 1e-3,
        epochs: int = 50,
        seed: int = 42,
        se: bool = True,
        devices="auto",
        recon_ckpt: str = "",
        recon_k: int = 3,
        recon_sum_to_one: bool = False,
):
    set_seed(seed)

    train_loader, val_loader = make_loaders(
        data_root, sensor_root, rgb, ir, batch_size=batch_size, num_workers=num_workers, val_ratio=0.2
    )

    # SPAN riceve 121 canali dall'HSI ricostruito
    model = LitReconThenSPAN(se=se, lr=lr, recon_ckpt=recon_ckpt, recon_k=recon_k, recon_sum_to_one=recon_sum_to_one)

    # callback
    ckpt = ModelCheckpoint(dirpath=save_dir, filename="best", monitor="val_loss", mode="min", save_top_k=1)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=100, min_delta=0.0, verbose=True)
    lrmon = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        max_epochs=epochs,
        accelerator="auto",
        devices=devices,
        precision="32-true",
        callbacks=[ckpt, early, lrmon],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--sensor_root", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="runs/span_lightning")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--se", type=bool, default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rgb", type=bool, default=False)
    ap.add_argument("--ir", type=bool, default=False)
    # nuovi argomenti
    ap.add_argument("--recon_ckpt", type=str, required=True, help="path ai pesi pre-addestrati di JointDualFilterMST (state_dict)")
    ap.add_argument("--recon_k", type=int, default=3)
    ap.add_argument("--recon_sum_to_one", action="store_true")
    args = ap.parse_args()

    main(
        data_root=args.data_root,
        sensor_root=args.sensor_root,
        rgb=args.rgb,
        ir=args.ir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        epochs=args.epochs,
        se=args.se,
        seed=args.seed,
        recon_ckpt=args.recon_ckpt,
        recon_k=args.recon_k,
        recon_sum_to_one=args.recon_sum_to_one,
    )
