from pathlib import Path
from torch.utils.data import DataLoader, random_split
from data_loader import *
from torch import nn, optim
import pytorch_lightning as pl
from models.our import *
from models.filter_opt import *


def make_loaders(root, sensor_root, rgb, ir, patch_mean, batch_size=8, num_workers=4, val_ratio=0.2, pin_memory=True):
    """
    Se esistono /train e /val sotto root li usa; altrimenti fa split random.
    """
    root = Path(root)
    full = FlourFolderDataset(root=root,
                              spectral_sens_csv=sensor_root,
                              rgb=rgb, ir=ir,
                              hsi_channels_first=False,  # True se i tuoi HSI sono (L,H,W)
                              illuminant_mode="planck",  # alogena
                              illuminant_T=2856.0,
                              patch_mean=patch_mean)
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
        x4, y = batch
        with torch.no_grad():  # recon congelato anche in train
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


class LitReconSpectral(pl.LightningModule):
    def __init__(self, spectral_sens_csv: str, recon_type: int,  lr: float = 1e-3, out_len: int = 121):
        super().__init__()
        self.save_hyperparameters()
        # NON fissare "cuda" qui: lascia che Lightning gestisca
        self.meas = DualFilterVector(spectral_sens_csv, device="cpu", dtype=torch.float32)
        self.dec = ResMLP8to121()

    def on_fit_start(self):
        # assicura coerenza di device/dtype con il trainer
        self.meas.to(self.device, dtype=self.dtype if hasattr(self, "dtype") else None)
        self.dec.to(self.device)

    def forward(self, s_true):
        y = self.meas(s_true)     # (B,8)
        s_pred = self.dec(y)      # (B,121)
        return s_pred, y


class FrozenFullRecon(nn.Module):
    """
    Carica l'intero modello di ricostruzione (meas + dec) da LitReconSpectral
    e congela tutti i parametri.
    """
    def __init__(self, ckpt_path: str, spectral_sens_csv: str, recon_type: int = 2):
        super().__init__()
        core = LitReconSpectral(
            spectral_sens_csv=spectral_sens_csv,
            recon_type=recon_type,
            lr=1e-3,
            out_len=121
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = core.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] load core: missing={missing}, unexpected={unexpected}")

        self.meas = core.meas   # DualFilterVector: 121 -> 8
        self.dec = core.dec     # decoder: 8 -> 121

        # congela tutto
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, s_true: torch.Tensor) -> torch.Tensor:
        """
        Esegue la pipeline completa: s_true [B,121] -> meas(121→8) -> dec(8→121)
        """
        if s_true.dim() == 4:
            s_true = s_true.mean(dim=(2, 3))  # media spaziale se input H×W
        y = self.meas(s_true)   # (B,8)
        s_pred = self.dec(y)    # (B,121)
        return s_pred


class SignalReconAndClassification(pl.LightningModule):
    """
    Pipeline completa:
      spettro [B,121] --[ DualFilterVector + decoder (frozen) ]--> HSI ricostruito [B,121]
                         --[ classificatore MLP ]--> logit (B,1)
    Loss: BCEWithLogits
    """
    def __init__(self,
                 recon_ckpt: str,
                 spectral_sens_csv: str,
                 lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        # 1) Ricostruttore completo (meas + dec)
        self.recon = FrozenFullRecon(
            ckpt_path=recon_ckpt,
            spectral_sens_csv=spectral_sens_csv
        )

        # 2) Classificatore (usa spettro ricostruito)
        self.model = MLPClassifier(input_dim=121, num_classes=1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr

    @torch.no_grad()
    def _reconstruct(self, s_true):
        return self.recon(s_true)

    def forward(self, s_true):
        # Ricostruzione e classificazione
        with torch.no_grad():
            s_pred = self._reconstruct(s_true)   # (B,121)
        logits = self.model(s_pred)              # (B,1)
        return logits

    def _step(self, batch, stage: str):
        x, y = batch                             # x: [B,121]
        with torch.no_grad():
            s_pred = self._reconstruct(x)
        logits = self.model(s_pred)

        y = y.long()
        loss = self.criterion(logits, y.float().unsqueeze(1))
        pred = (logits.view(-1) >= 0.0).long()
        acc = (pred == y.view(-1)).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=50
        )
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
        self.recon.to(self.device)
