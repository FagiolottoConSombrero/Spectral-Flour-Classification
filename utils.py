from pathlib import Path
import math
from torch.utils.data import DataLoader, random_split
from data_loader import *
from torch import nn, optim
import pytorch_lightning as pl
from models.our import *
from models.filter_opt import *


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
    def __init__(self, se: bool, lr: float, recon_ckpt: str, sensor_root, recon_k: int = 3, recon_sum_to_one: bool = False, ):
        super().__init__()
        self.save_hyperparameters()

        # 1) Ricostruttore (congelato)
        self.recon = JointDualFilterMST(sensor_root)
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
        in_channels = 8
        self.model = MLPClassifier(input_dim=121, num_classes=1)

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
            hsi = self._reconstruct_hsi(x4, return_x8=True)
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