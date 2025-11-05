import random
import argparse
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from utils import *


# ---------------- util ----------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    model = LitReconThenSPAN(se=se, lr=lr, recon_ckpt=recon_ckpt, recon_k=recon_k, recon_sum_to_one=recon_sum_to_one, sensor_root=sensor_root)

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

    arg = argparse.ArgumentParser()
    arg.add_argument("--data_root", type=str, required=True)
    arg.add_argument("--sensor_root", type=str, required=True)
    arg.add_argument("--save_dir", type=str, default="runs/span_lightning")
    arg.add_argument("--batch_size", type=int, default=8)
    arg.add_argument("--num_workers", type=int, default=4)
    arg.add_argument("--lr", type=float, default=1e-4)
    arg.add_argument("--epochs", type=int, default=50)
    arg.add_argument("--se", type=bool, default=True)
    arg.add_argument("--seed", type=int, default=42)
    arg.add_argument("--rgb", type=bool, default=False)
    arg.add_argument("--ir", type=bool, default=False)
    # nuovi argomenti
    arg.add_argument("--recon_ckpt", type=str, required=True, help="path ai pesi pre-addestrati di JointDualFilterMST (state_dict)")
    arg.add_argument("--recon_k", type=int, default=3)
    arg.add_argument("--recon_sum_to_one", action="store_true")
    args = arg.parse_args()

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
