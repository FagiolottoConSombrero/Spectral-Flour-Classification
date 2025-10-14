import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class FlourFolderDataset(Dataset):
    """
    Legge .h5 organizzati per classi in sottocartelle:
    root/class_name/*.h5
    Ritorna (x, y) con y = indice della classe (class_to_idx ordinato alfabeticamente).
    """
    def __init__(self, root, dataset_keys=("image", "data"), exclude_prefixes=("._",), dtype=torch.float32):
        self.root = root
        self.dataset_keys = dataset_keys
        self.exclude_prefixes = exclude_prefixes
        self.dtype = dtype

        # trova le classi (sottocartelle)
        self.classes = sorted([d for d in os.listdir(root)
                               if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")])
        if not self.classes:
            raise RuntimeError(f"Nessuna classe trovata in {root}")
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # indicizza i file (path, label_idx)
        samples = []
        for cls in self.classes:
            cdir = os.path.join(root, cls)
            for fname in sorted(os.listdir(cdir)):
                if not fname.endswith(".h5"):
                    continue
                if any(fname.startswith(pfx) for pfx in self.exclude_prefixes):
                    continue
                fpath = os.path.join(cdir, fname)
                samples.append((fpath, self.class_to_idx[cls]))

        if not samples:
            raise RuntimeError(f"Nessun .h5 trovato sotto {root}")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        x = self._load_h5(path)  # torch.Tensor
        return x, y

    def _load_h5(self, path):
        with h5py.File(path, "r") as f:
            arr = None
            # prova i dataset key in ordine
            for k in self.dataset_keys:
                if k in f:
                    arr = f[k][()]
                    break
            if arr is None:
                # fallback: primo dataset trovato
                for k in f.keys():
                    if isinstance(f[k], h5py.Dataset):
                        arr = f[k][()]
                        break
            if arr is None:
                raise KeyError(f"Nessun dataset valido in {path}")

        # arr -> torch.Tensor float32, (C,H,W)
        if arr.ndim != 3:
            raise ValueError(f"atteso 3D, trovato {arr.shape} in {path}")

        # Se è (H,W,C) portalo a (C,H,W)
        if arr.shape[0] not in (121, 3) and arr.shape[-1] in (121, 3):
            arr = np.moveaxis(arr, -1, 0)

        x = torch.from_numpy(arr).to(self.dtype)

        # assicurati (C,H,W)
        if x.dim() != 3:
            raise ValueError(f"forma non valida dopo conversione: {tuple(x.shape)}")
        if x.shape[0] not in (121, 1, 3) and x.shape[0] not in range(1, 2049):
            # controllo blando, puoi rimuoverlo se vuoi
            pass
        return x

