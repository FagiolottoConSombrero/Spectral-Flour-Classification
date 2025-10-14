import argparse
from pathlib import Path
import h5py
import numpy as np
import sys

def load_first_dataset(h5_path: Path, dataset_name: str | None):
    with h5py.File(h5_path, "r") as f:
        if dataset_name is not None:
            if dataset_name not in f:
                raise KeyError(f"Dataset '{dataset_name}' non trovato in {h5_path.name}. "
                               f"Disponibili: {list(f.keys())}")
            data = f[dataset_name][:]
        else:
            # Prendi il primo dataset “foglia”
            def first_leaf(obj):
                for k, v in obj.items():
                    if isinstance(v, h5py.Dataset):
                        return v[...]
                    elif isinstance(v, h5py.Group):
                        res = first_leaf(v)
                        if res is not None:
                            return res
                return None
            data = first_leaf(f)
            if data is None:
                raise ValueError(f"Nessun dataset trovato in {h5_path.name}.")
    return data

def save_patch(out_path: Path, patch: np.ndarray, dataset_name: str = "image"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset(dataset_name, data=patch, compression="gzip", compression_opts=4)

def extract_patches(arr: np.ndarray, patch_h: int, patch_w: int):
    """
    arr: atteso (H, W, C) = (224, 224, 121)
    Ritorna lista di patch non sovrapposte (ph, pw, C).
    """
    if arr.ndim != 3:
        raise ValueError(f"L'array deve essere 3D (H, W, C), trovato {arr.shape}")

    H, W, C = arr.shape
    if (H, W) != (224, 224):
        print(f"[ATTENZIONE] Immagine con dimensioni {H}x{W} anziché 224x224. Procedo comunque.", file=sys.stderr)
    if C != 121:
        print(f"[ATTENZIONE] Numero di canali {C} anziché 121. Procedo comunque.", file=sys.stderr)

    if H % patch_h != 0 or W % patch_w != 0:
        raise ValueError(
            f"La patch ({patch_h}x{patch_w}) deve dividere perfettamente l'immagine ({H}x{W})."
        )

    patches = []
    for y in range(0, H, patch_h):
        for x in range(0, W, patch_w):
            patches.append(arr[y:y+patch_h, x:x+patch_w, :])
    return patches

def process_file(h5_path: Path, out_dir: Path, patch_h: int, patch_w: int, dataset_in: str | None, dataset_out: str):
    img = load_first_dataset(h5_path, dataset_in)

    # Se l'ordine fosse (C, H, W) o simili, prova ad adattare: usa le due dimensioni uguali come H e W
    if img.ndim == 3 and img.shape[0] == 224 and img.shape[1] == 224:
        # (H, W, C) già a posto
        pass
    elif img.ndim == 3 and img.shape[-2] == 224 and img.shape[-1] == 224:
        # Probabile (C, H, W) -> porta a (H, W, C)
        img = np.moveaxis(img, 0, -1)
    elif img.ndim == 3 and img.shape[0] == 224 and img.shape[-1] == 224:
        # (H, C, W) -> (H, W, C)
        img = np.moveaxis(img, 1, -1)
    else:
        # Tenta ultima chance: trova due dimensioni uguali e spostale in testa
        HWC_candidates = None
        shp = img.shape
        for i in range(3):
            for j in range(3):
                if i != j and shp[i] == 224 and shp[j] == 224:
                    k = 3 - i - j
                    img = np.moveaxis(img, (i, j, k), (0, 1, 2))
                    HWC_candidates = img
                    break
            if HWC_candidates is not None:
                break
        if HWC_candidates is None:
            raise ValueError(f"Forma non riconosciuta: {img.shape}. Atteso 224x224 su due assi.")

    patches = extract_patches(img, patch_h, patch_w)

    base = h5_path.stem
    idx = 0
    for y in range(0, img.shape[0], patch_h):
        for x in range(0, img.shape[1], patch_w):
            patch = img[y:y+patch_h, x:x+patch_w, :]
            out_name = f"{base}_y{y:03d}_x{x:03d}_ph{patch_h}_pw{patch_w}.h5"
            save_patch(out_dir / out_name, patch, dataset_name=dataset_out)
            idx += 1
    return idx

def main():
    parser = argparse.ArgumentParser(
        description="Estrarre patch non sovrapposte uguali da immagini 224x224x121 in file .h5"
    )
    parser.add_argument("--input_dir", type=Path, help="Cartella con i .h5")
    parser.add_argument("--output_dir", type=Path, help="Cartella di destinazione per le patch .h5")
    parser.add_argument("--patch-size", type=int, default=56,
                        help="Lato della patch quadrata (deve dividere 224). Esempi: 16, 28, 32, 56, 112.")
    parser.add_argument("--patch-width", type=int, default=None,
                        help="(Opzionale) larghezza patch; se non dato, usa --patch-size (patch quadrata)")
    parser.add_argument("--dataset-in", type=str, default=None,
                        help="Nome del dataset da leggere nell'h5 (se non fornito, usa il primo dataset trovato).")
    parser.add_argument("--dataset-out", type=str, default="patch",
                        help="Nome del dataset da scrivere nelle patch in output.")
    args = parser.parse_args()

    patch_h = args.patch_size
    patch_w = args.patch_width if args.patch_width is not None else args.patch_size

    if 224 % patch_h != 0 or 224 % patch_w != 0:
        raise SystemExit(f"Errore: la patch ({patch_h}x{patch_w}) deve dividere 224 senza resto.")

    h5_files = sorted([p for p in args.input_dir.glob("*.h5") if p.is_file()])
    if not h5_files:
        raise SystemExit(f"Nessun file .h5 trovato in {args.input_dir}")

    total_patches = 0
    for fpath in h5_files:
        try:
            n = process_file(fpath, args.output_dir, patch_h, patch_w, args.dataset_in, args.dataset_out)
            print(f"[OK] {fpath.name}: {n} patch")
            total_patches += n
        except Exception as e:
            print(f"[ERRORE] {fpath.name}: {e}", file=sys.stderr)

    print(f"Fatto. Patch totali generate: {total_patches}")


if __name__ == "__main__":
    main()
