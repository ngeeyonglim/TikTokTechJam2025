#!/usr/bin/env python3
"""
Split a YOLOv8 dataset (images/ + labels/) into N client shards.

Features
- Recursively scans images/ (jpg/jpeg/png/bmp/tif/tiff)
- Pairs each image with its YOLO label (mirrors relative path under labels/)
- Deterministic partition into num_clients shards
- Per-client train/val split
- Optional subsampling per client:
    --client-frac <0..1> to keep a fraction
    --client-max  <int>  to cap the count (applied after frac)
- Symlink by default (fast, saves space) or --copy to copy files
- Creates client_XX.yaml usable by Ultralytics
- Optionally creates empty labels for images missing annotations

Python 3.8+ (tested on 3.13).
"""

import argparse
import errno
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# --- ADD/REPLACE in your script ---------------------------------------------
COCO_NAMES_WITH_FACE = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
    80: "face",  # your added class id
}

def _gather_images_under(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])

def _write_split_lists(client_dir: Path) -> Tuple[Path, Path]:
    """
    Create train.txt and val.txt under client_dir, each line a RELATIVE path to an image
    from client_dir (e.g., 'images/train/sub/xxx.jpg').
    """
    train_imgs = _gather_images_under(client_dir / "images/train")
    val_imgs = _gather_images_under(client_dir / "images/val")

    def rel_lines(imgs: List[Path]) -> List[str]:
        return [str(p.relative_to(client_dir).as_posix()) for p in imgs]

    train_txt = client_dir / "train.txt"
    val_txt = client_dir / "val.txt"
    train_txt.write_text("\n".join(rel_lines(train_imgs)) + ("\n" if train_imgs else ""))
    val_txt.write_text("\n".join(rel_lines(val_imgs)) + ("\n" if val_imgs else ""))
    return train_txt, val_txt

def write_yaml_coco_style(out_root: Path, client_idx: int, names: Dict[int, str]) -> Path:
    """
    Emit a YOLOv8-style YAML:
      path: <client_dir>
      train: images/train
      val: images/val
      names: {0: ..., 80: face}
    """
    import yaml
    client_dir = out_root / f"client_{client_idx:02d}"
    yaml_path = out_root / f"client_{client_idx:02d}.yaml"

    data = {
        "path": str(client_dir),
        "train": "images/train",
        "val": "images/val",
        "names": names,
        "nc": 81,
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=True)
    return yaml_path
# --- END ADDITIONS -----------------------------------------------------------



def find_label_for_image(img_path: Path, images_dir: Path, labels_dir: Path) -> Optional[Path]:
    """
    Prefer a mirror relative path inside labels/, else fall back to basename match.
    """
    rel = img_path.relative_to(images_dir)
    label_rel = rel.with_suffix(".txt")
    candidate = labels_dir / label_rel
    if candidate.exists():
        return candidate

    # Fallback: search by basename (slower if many files)
    basename = img_path.stem + ".txt"
    matches = list(labels_dir.rglob(basename))
    if len(matches) == 1:
        return matches[0]
    return None


def collect_pairs(
    images_dir: Path,
    labels_dir: Path,
    create_empty_missing: bool = False,
) -> Tuple[List[Tuple[Path, Optional[Path]]], List[Path]]:
    imgs = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs.sort()
    pairs: List[Tuple[Path, Optional[Path]]] = []
    missing: List[Path] = []
    for img in imgs:
        lab = find_label_for_image(img, images_dir, labels_dir)
        if lab is None:
            missing.append(img)
            pairs.append((img, None))
        else:
            pairs.append((img, lab))
    if create_empty_missing and missing:
        print(f"[INFO] {len(missing)} images missing labels (will create empty labels in OUTPUT).")
    elif missing:
        print(
            f"[WARN] {len(missing)} images missing labels. They will be included without a label file.\n"
            f"       Pass --create-empty-missing to create empty .txt in outputs."
        )
    return pairs, missing


def split_indices(n: int, num_clients: int, seed: int) -> List[List[int]]:
    """
    Deterministic partition into num_clients nearly equal parts.
    """
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)

    chunks: List[List[int]] = []
    base = n // num_clients
    rem = n % num_clients
    start = 0
    for i in range(num_clients):
        size = base + (1 if i < rem else 0)
        chunks.append(idxs[start : start + size])
        start += size
    return chunks


def train_val_split(indices: List[int], val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    idxs = indices[:]
    rng.shuffle(idxs)
    val_sz = int(round(len(idxs) * val_frac))
    val_idx = set(idxs[:val_sz])
    train = [i for i in idxs if i not in val_idx]
    val = [i for i in idxs if i in val_idx]
    return train, val


def safe_symlink(src: Path, dst: Path) -> None:
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    except Exception:
        # Fallback to copy if symlink fails on this FS
        shutil.copy2(src, dst)


def copy_or_link(src: Path, dst: Path, do_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_copy:
        shutil.copy2(src, dst)
    else:
        safe_symlink(src, dst)


def write_yaml(out_root: Path, client_idx: int, names: List[str] = ["face"]) -> Path:
    client_dir = out_root / f"client_{client_idx:02d}"
    yaml_path = out_root / f"client_{client_idx:02d}.yaml"
    content = (
        f"path: {client_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names: {names}\n"
    )
    yaml_path.write_text(content)
    return yaml_path


def main():
    ap = argparse.ArgumentParser(
        description="Split YOLOv8 dataset into N client datasets with optional subsampling and train/val splits."
    )
    ap.add_argument("--images-dir", required=True, type=Path, help="Path to source images/ directory")
    ap.add_argument("--labels-dir", required=True, type=Path, help="Path to source labels/ directory (YOLO .txt)")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output root directory for client shards")
    ap.add_argument("--num-clients", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction per client (0..1)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks")
    ap.add_argument(
        "--create-empty-missing",
        action="store_true",
        help="If a label is missing, create an empty label file in the OUTPUT shard.",
    )
    # NEW: subsampling controls
    ap.add_argument(
        "--client-frac",
        type=float,
        default=1.0,
        help="Fraction of each client's pool to keep BEFORE train/val (e.g., 0.5 keeps half).",
    )
    ap.add_argument(
        "--client-max",
        type=int,
        default=0,
        help="If >0, cap images per client AFTER applying --client-frac.",
    )
    args = ap.parse_args()

    images_dir = args.images_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    out_root = args.out_dir.resolve()

    # Create output root, with read-only (/data) fallback to HOME if needed
    try:
        out_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if e.errno == errno.EROFS:  # Read-only FS (e.g., macOS root /data)
            fallback = Path.home() / out_root.name
            print(f"[WARN] '{out_root}' is read-only. Falling back to '{fallback}'.")
            fallback.mkdir(parents=True, exist_ok=True)
            out_root = fallback
        else:
            raise

    # Sanity checks
    if not images_dir.exists():
        print(f"[ERROR] images-dir not found: {images_dir}", file=sys.stderr)
        sys.exit(1)
    if not labels_dir.exists():
        print(f"[ERROR] labels-dir not found: {labels_dir}", file=sys.stderr)
        sys.exit(1)

    # Gather (image, label) pairs
    pairs, missing = collect_pairs(images_dir, labels_dir, create_empty_missing=args.create_empty_missing)
    n = len(pairs)
    print(f"[INFO] Found {n} images in {images_dir}")
    if n == 0:
        print("[ERROR] No images found. Check --images-dir.", file=sys.stderr)
        sys.exit(1)

    print(
        f"[INFO] num_clients={args.num_clients}, val_frac={args.val_frac:.2f}, "
        f"seed={args.seed}, client_frac={args.client_frac}, client_max={args.client_max}"
    )

    # Partition across clients
    client_chunks = split_indices(n, args.num_clients, args.seed)

    summary: Dict[int, Dict[str, int]] = {}

    for ci, idxs in enumerate(client_chunks):
        # --- Subsample each client's pool BEFORE train/val ---
        keep = idxs[:]  # start from this client's pool
        if args.client_frac < 1.0 or args.client_max > 0:
            rng = random.Random(args.seed + 10_000 + ci)  # deterministic per client
            k = len(keep)
            # apply fractional keep
            if args.client_frac < 1.0:
                k = max(1, int(math.ceil(k * args.client_frac)))
            # apply absolute cap
            if args.client_max > 0:
                k = min(k, args.client_max)
            # sample without replacement
            keep = rng.sample(keep, k)
        # -----------------------------------------------------

        # Train/val split on the reduced pool
        tr_idx, va_idx = train_val_split(keep, args.val_frac, seed=args.seed + ci)

        client_dir = out_root / f"client_{ci:02d}"
        img_train_dir = client_dir / "images/train"
        img_val_dir = client_dir / "images/val"
        lab_train_dir = client_dir / "labels/train"
        lab_val_dir = client_dir / "labels/val"

        def place(i: int, img_dst_dir: Path, lab_dst_dir: Path):
            img_src, lab_src = pairs[i]
            rel = img_src.relative_to(images_dir)
            img_dst = img_dst_dir / rel
            copy_or_link(img_src, img_dst, args.copy)

            lab_rel = rel.with_suffix(".txt")
            lab_dst = lab_dst_dir / lab_rel
            if lab_src and lab_src.exists():
                copy_or_link(lab_src, lab_dst, args.copy)
            else:
                if args.create_empty_missing:
                    lab_dst.parent.mkdir(parents=True, exist_ok=True)
                    lab_dst.write_text("")  # empty file allowed by YOLO
                # else: skip creating label file

        for i in tr_idx:
            place(i, img_train_dir, lab_train_dir)
        for i in va_idx:
            place(i, img_val_dir, lab_val_dir)

        # YAML per client
        yaml_path = write_yaml_coco_style(out_root, ci, COCO_NAMES_WITH_FACE)


        summary[ci] = {"total": len(keep), "train": len(tr_idx), "val": len(va_idx)}
        print(
            f"[CLIENT {ci:02d}] kept={len(keep)}  train={len(tr_idx)}  val={len(va_idx)}"
            f"  -> {yaml_path}"
        )

    # Log missing labels (if any)
    if missing:
        log = out_root / "missing_labels.log"
        with log.open("w") as f:
            for p in missing:
                f.write(str(p) + "\n")
        print(f"[WARN] Wrote list of {len(missing)} images missing labels: {log}")

    # Global summary
    tot_keep = sum(v["total"] for v in summary.values())
    tot_tr = sum(v["train"] for v in summary.values())
    tot_va = sum(v["val"] for v in summary.values())
    print(f"\n[SUMMARY] kept={tot_keep}  train={tot_tr}  val={tot_va}  clients={args.num_clients}")
    print(f"[DONE] Output root: {out_root}")


if __name__ == "__main__":
    main()
