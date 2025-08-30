#!/usr/bin/env python3
"""
Create N client datasets where each client gets an equal mix:
half from OLD root and half from NEW root.

Both roots must have:
  <root>/images/...
  <root>/labels/...

Per client we also keep that old/new balance inside train/val.
Symlink by default (fast); use --copy to copy files instead.
"""

import argparse
import errno
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------- config -----------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

COCO_NAMES_WITH_FACE = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush", 80: "face",
}

# -------------- helpers -------------------

@dataclass(frozen=True)
class Pair:
    img: Path
    lbl: Optional[Path]
    rel: Path           # relative path under <root>/images/


def _gather_images(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _find_label_for_rel(rel_img: Path, labels_dir: Path) -> Optional[Path]:
    """
    Given a relative image path (e.g., 'sub/a.jpg'), map to labels/sub/a.txt.
    If not found, try basename search under labels_dir.
    """
    cand = labels_dir / rel_img.with_suffix(".txt")
    if cand.exists():
        return cand
    # fallback basename search (slower)
    matches = list(labels_dir.rglob(rel_img.stem + ".txt"))
    if len(matches) == 1:
        return matches[0]
    return None


def collect_pairs_under_root(root: Path, create_empty_missing_log: bool = False) -> Tuple[List[Pair], List[Path]]:
    """
    Scan <root>/images and pair with <root>/labels by mirroring relative path.
    Returns (pairs, missing_images_with_no_label)
    """
    images_dir = root / "images"
    labels_dir = root / "labels"
    imgs = _gather_images(images_dir)
    pairs: List[Pair] = []
    missing: List[Path] = []

    for img in imgs:
        rel = img.relative_to(images_dir)
        lbl = _find_label_for_rel(rel, labels_dir)
        if lbl is None:
            missing.append(img)
        pairs.append(Pair(img=img, lbl=lbl, rel=rel))

    if not pairs:
        print(f"[ERROR] No images found under {images_dir} (looked for {sorted(IMG_EXTS)})", file=sys.stderr)
    if missing and create_empty_missing_log:
        print(f"[INFO] {len(missing)} images missing labels under {root} (will create empty labels in outputs).")
    elif missing:
        print(f"[WARN] {len(missing)} images missing labels under {root}. They will be included without a label file.")
    return pairs, missing


def safe_symlink(src: Path, dst: Path):
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    except Exception:
        shutil.copy2(src, dst)


def copy_or_link(src: Path, dst: Path, do_copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if do_copy:
        shutil.copy2(src, dst)
    else:
        safe_symlink(src, dst)


def train_val_split(items: List[Pair], val_frac: float, seed: int) -> Tuple[List[Pair], List[Pair]]:
    rng = random.Random(seed)
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    v = int(round(len(items) * val_frac))
    val_idx = set(idxs[:v])
    tr = [items[i] for i in idxs if i not in val_idx]
    va = [items[i] for i in idxs if i in val_idx]
    return tr, va


def write_yaml_coco_style(out_root: Path, client_idx: int, names: Dict[int, str]) -> Path:
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
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=True)
    return yaml_path

# -------------- main logic ----------------

def split_equal_old_new(
    old_root: Path,
    new_root: Path,
    out_root: Path,
    num_clients: int,
    per_client_total: Optional[int],
    val_frac: float,
    seed: int,
    copy_files: bool,
    create_empty_missing: bool,
):
    # collect from both roots
    old_pairs, old_missing = collect_pairs_under_root(old_root, create_empty_missing)
    new_pairs, new_missing = collect_pairs_under_root(new_root, create_empty_missing)

    if not old_pairs:
        raise ValueError(f"No YOLO pairs found under OLD root: {old_root}")
    if not new_pairs:
        raise ValueError(f"No YOLO pairs found under NEW root: {new_root}")

    # shuffle deterministically
    rng = random.Random(seed)
    rng.shuffle(old_pairs)
    rng.shuffle(new_pairs)

    # compute per-source quota per client
    max_equal_per_client = min(len(old_pairs), len(new_pairs)) // num_clients
    if max_equal_per_client == 0:
        raise ValueError("Not enough data in one of the roots to split across clients equally.")

    if per_client_total is None:
        per_source = max_equal_per_client
    else:
        if per_client_total % 2 != 0:
            raise ValueError("--per-client-total must be even (half old + half new).")
        req_per_source = per_client_total // 2
        if req_per_source > max_equal_per_client:
            raise ValueError(
                f"Requested per-client-total={per_client_total} exceeds feasible maximum "
                f"{2*max_equal_per_client} (limited by available old/new samples)."
            )
        per_source = req_per_source

    print(f"[SPLIT] clients={num_clients} | per-client total={2*per_source} (old={per_source}, new={per_source}) "
          f"| val_frac={val_frac:.2f}")

    # slice contiguous chunks for determinism
    def chunk(seq: List[Pair], k: int, size: int) -> List[List[Pair]]:
        return [seq[i*size:(i+1)*size] for i in range(k)]

    old_chunks = chunk(old_pairs, num_clients, per_source)
    new_chunks = chunk(new_pairs, num_clients, per_source)

    # prepare out root
    try:
        out_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if e.errno == errno.EROFS:
            fallback = Path.home() / out_root.name
            print(f"[WARN] '{out_root}' is read-only. Falling back to '{fallback}'.")
            fallback.mkdir(parents=True, exist_ok=True)
            out_root = fallback
        else:
            raise

    # build each client
    totals = {"train": 0, "val": 0}
    for ci in range(num_clients):
        client_dir = out_root / f"client_{ci:02d}"
        img_tr = client_dir / "images/train"
        img_va = client_dir / "images/val"
        lab_tr = client_dir / "labels/train"
        lab_va = client_dir / "labels/val"

        # split each source, then merge (keeps balance in train/val)
        old_tr, old_va = train_val_split(old_chunks[ci], val_frac, seed + 101 + ci)
        new_tr, new_va = train_val_split(new_chunks[ci], val_frac, seed + 202 + ci)

        train_pairs = old_tr + new_tr
        val_pairs   = old_va + new_va

        # shuffle merged sets so old/new are interleaved
        rng.shuffle(train_pairs)
        rng.shuffle(val_pairs)

        def place(pair: Pair, img_dst_root: Path, lab_dst_root: Path):
            img_dst = img_dst_root / pair.rel
            copy_or_link(pair.img, img_dst, copy_files)
            lab_rel = pair.rel.with_suffix(".txt")
            lab_dst = lab_dst_root / lab_rel
            if pair.lbl and pair.lbl.exists():
                copy_or_link(pair.lbl, lab_dst, copy_files)
            else:
                if create_empty_missing:
                    lab_dst.parent.mkdir(parents=True, exist_ok=True)
                    lab_dst.write_text("")  # empty file allowed by YOLO

        for p in train_pairs:
            place(p, img_tr, lab_tr)
        for p in val_pairs:
            place(p, img_va, lab_va)

        yaml_path = write_yaml_coco_style(out_root, ci, COCO_NAMES_WITH_FACE)
        print(f"[CLIENT {ci:02d}] train={len(train_pairs)}  val={len(val_pairs)}  -> {yaml_path}")

        totals["train"] += len(train_pairs)
        totals["val"]   += len(val_pairs)

    # log missing labels (optional)
    missing_total = len(old_missing) + len(new_missing)
    if missing_total:
        log = out_root / "missing_labels.log"
        with log.open("w") as f:
            for p in old_missing + new_missing:
                f.write(str(p) + "\n")
        print(f"[WARN] Wrote list of {missing_total} images missing labels: {log}")

    print(f"\n[SUMMARY] total train={totals['train']}  total val={totals['val']}  clients={num_clients}")
    print(f"[DONE] Output root: {out_root}")


def parse_args():
    ap = argparse.ArgumentParser(description="Build equal OLD/NEW mixed client datasets for YOLOv8.")
    ap.add_argument("--old-root", required=True, type=Path, help="OLD dataset root (has images/ and labels/)")
    ap.add_argument("--new-root", required=True, type=Path, help="NEW dataset root (has images/ and labels/)")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory for client_* datasets")
    ap.add_argument("--num-clients", type=int, default=4)
    ap.add_argument("--per-client-total", type=int, default=None,
                    help="Total images per client (must be even). Default: maximum feasible.")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction per client (0..1)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlink")
    ap.add_argument("--create-empty-missing", action="store_true",
                    help="Create empty label files when a source image has no label.")
    return ap.parse_args()


def main():
    args = parse_args()
    old_root = args.old_root.resolve()
    new_root = args.new_root.resolve()

    for r in (old_root, new_root):
        if not (r / "images").exists() or not (r / "labels").exists():
            print(f"[ERROR] {r} must contain 'images/' and 'labels/'", file=sys.stderr)
            sys.exit(1)

    split_equal_old_new(
        old_root=old_root,
        new_root=new_root,
        out_root=args.out_dir.resolve(),
        num_clients=args.num_clients,
        per_client_total=args.per_client_total,
        val_frac=args.val_frac,
        seed=args.seed,
        copy_files=args.copy,
        create_empty_missing=args.create_empty_missing,
    )


if __name__ == "__main__":
    main()
