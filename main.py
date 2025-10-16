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


# ---------------- util ----------------
def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def infer_in_channels(loader: DataLoader) -> int:
    for xb, _ in loader:
        return int(xb.shape[1])
    raise RuntimeError("Dataloader vuoto: impossibile inferire i canali.")

def make_loaders(root, batch_size=8, num_workers=4, val_ratio=0.2, pin_memory=True):
    """
    Se esistono /train e /val sotto root li usa; altrimenti fa split random.
    """
    root = Path(root)

    if (root / "train").is_dir() and (root / "val").is_dir():
        train_set = FlourFolderDataset(root / "train")
        val_set   = FlourFolderDataset(root / "val")
    else:
        full = FlourFolderDataset(root)
        n = len(full)
        n_val  = int(math.floor(n * val_ratio))
        n_train = n - n_val
        train_set, val_set = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    pw = num_workers > 0

    def _collate(batch):
        xs, ys = zip(*batch)
        xs = torch.stack(xs, 0)             # (B,C,H,W)
        ys = torch.as_tensor(ys, dtype=torch.long)
        return xs, ys

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=pw,
        collate_fn=_collate
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=pw,
        collate_fn=_collate
    )
    return train_loader, val_loader


# ------------- LightningModule -------------
class LitSPANBinary(pl.LightningModule):
    """
    Classificazione binaria: SPAN emette probabilità (sigmoid) shape (B,1).
    Loss: BCELoss. Metric: accuracy (soglia 0.5).
    """
    def __init__(self, in_channels: int, se: bool, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = SPAN(num_in_ch=in_channels, feature_channels=48, bias=True, se=se)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)  # (B,1) probabilità

    def _step(self, batch, stage: str):
        x, y = batch
        y_int = y.long()
        y_f = y_int.float().unsqueeze(1)
        probs = self(x)
        loss = self.criterion(probs, y_f)
        pred = (probs.view(-1) >= 0.5).long()
        acc = (pred == y_int.view(-1)).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.hparams.lr)
        # riduce LR quando la val_loss non migliora (patience 30)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=50)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


# ---------------- main ----------------
def main(
    data_root: str,
    save_dir: str = "runs/span_lightning",
    batch_size: int = 8,
    num_workers: int = 4,
    lr: float = 1e-3,
    epochs: int = 50,
    seed: int = 42,
    se: bool = True,
    devices="auto",
):
    set_seed(seed)

    train_loader, val_loader = make_loaders(
        data_root, batch_size=batch_size, num_workers=num_workers, val_ratio=0.2
    )

    # inferisci C dai dati
    in_ch = infer_in_channels(DataLoader(train_loader.dataset, batch_size=1, shuffle=False))
    model = LitSPANBinary(in_channels=in_ch, se=se, lr=lr)

    # checkpoint + early stopping (stop se val_loss non migliora per 100 epoche)
    ckpt = ModelCheckpoint(dirpath=save_dir, filename="best", monitor="val_loss", mode="min", save_top_k=1)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=100, min_delta=0.0, verbose=True)
    lrmon = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        max_epochs=epochs,
        accelerator="auto",
        devices=devices,
        precision="32-true",  # sicuro sulla TITAN Xp
        callbacks=[ckpt, early, lrmon],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="runs/span_lightning")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--se", type=bool, default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    main(
        data_root=args.data_root,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        epochs=args.epochs,
        se=args.se,
        seed=args.seed,
    )

