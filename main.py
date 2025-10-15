import math, random
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR

# ====== riusa le tue classi già definite ======
# from your_module import FlourFolderDataset, SPAN
# Se sono nello stesso file, rimuovi questi import.
from data_loader import FlourFolderDataset
from models.our import SPAN


def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_loaders(root, batch_size=8, num_workers=4, val_ratio=0.2, test_ratio=0.0, pin_memory=True):
    root = Path(root)
    # se esistono split su disco, usali
    if (root / "train").is_dir() and (root / "val").is_dir():
        train_set = FlourFolderDataset(root / "train")
        val_set = FlourFolderDataset(root / "val")
        test_set = FlourFolderDataset(root / "test") if (root / "test").is_dir() else None
    else:
        full = FlourFolderDataset(root)
        n = len(full)
        n_test = int(math.floor(n * test_ratio))
        n_val  = int(math.floor(n * val_ratio))
        n_train = n - n_val - n_test
        splits = [n_train, n_val] if n_test == 0 else [n_train, n_val, n_test]
        parts = random_split(full, splits, generator=torch.Generator().manual_seed(0))
        train_set, val_set = parts[0], parts[1]
        test_set = parts[2] if len(parts) == 3 else None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    return train_loader, val_loader, test_loader


class LitSPANBinary(pl.LightningModule):
    """
    Usa SPAN come classificatore binario che emette probs sigmoid (B,1).
    Loss: BCELoss. Metric: accuracy binaria (soglia 0.5).
    """
    def __init__(self, in_channels, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = SPAN(num_in_ch=in_channels, feature_channels=48, bias=True)
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.model(x)  # (B,1) probs

    def _step_common(self, batch, stage: str):
        x, y = batch
        y_int = y.long()
        y_f = y_int.float().unsqueeze(1)
        probs = self(x)
        loss = self.criterion(probs, y_f)
        pred = (probs.view(-1) >= 0.5).long()
        acc = (pred == y_int.view(-1)).float().mean()
        # log su progress bar ed epoch
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step_common(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step_common(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step_common(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=30)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }


def infer_in_channels(loader):
    for xb, _ in loader:
        return xb.shape[1]
    raise RuntimeError("Dataloader vuoto: impossibile inferire i canali.")


def main(data_root, save_dir="", batch_size=8, num_workers=4, lr=1e-3, epochs=10, seed=42, devices="auto"):
    set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(
        data_root, batch_size=batch_size, num_workers=num_workers, val_ratio=0.2, test_ratio=0.0
    )
    in_ch = infer_in_channels(DataLoader(train_loader.dataset, batch_size=1, shuffle=False))
    model = LitSPANBinary(in_channels=in_ch, lr=lr)

    # checkpoint sul best val_loss (minimo)
    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir, filename="best", monitor="val_loss", mode="min", save_top_k=1
    )

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        max_epochs=epochs,
        devices=devices,
        accelerator="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[ckpt],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)  # fit = train+val
    # test sul best checkpoint se disponibile
    ckpt_path = ckpt.best_model_path if ckpt.best_model_path else None
    if test_loader is not None:
        trainer.test(dataloaders=test_loader, ckpt_path=ckpt_path)  # usa test_step() definito sopra
    else:
        print("Nessun test set: salto il test.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="runs/span_lightning")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(
        data_root=args.data_root,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )
