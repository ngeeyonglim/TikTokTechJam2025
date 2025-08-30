#!/usr/bin/env python3
import argparse
import sys
import yaml
from pathlib import Path
from typing import List, Union
from ultralytics import YOLO

def _load_yaml(p: Path):
    with p.open() as f:
        return yaml.safe_load(f)

def _save_yaml(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def _to_list(x: Union[str, List[str]]):
    if isinstance(x, list):
        return x
    return [x]

def sanity_check_81(data_yaml_path: Path):
    d = _load_yaml(data_yaml_path)

    # Required keys
    for k in ("path", "train", "val", "names", "nc"):
        if k not in d:
            raise ValueError(f"'{k}' missing in {data_yaml_path}")

    # Normalize names (dict -> ordered list)
    names = d["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
        d["names"] = names

    if d["nc"] != 81 or len(d["names"]) != 81:
        raise ValueError(f"Expected 81 classes. nc={d['nc']}, len(names)={len(d['names'])}")

    # Quick check: last class should be 'face' at index 80
    if d["names"][80] != "face":
        print("[WARN] names[80] is not 'face'. Make sure your labels use class id 80 for face.", file=sys.stderr)

    # Existence checks (relative to path if relative paths)
    root = Path(d["path"])
    for split in ("train", "val"):
        split_item = d[split]
        if isinstance(split_item, list):
            for item in split_item:
                p = (root / item) if not str(item).startswith("/") else Path(item)
                if not p.exists():
                    raise FileNotFoundError(f"{split} item not found: {p}")
        else:
            p = (root / split_item) if not str(split_item).startswith("/") else Path(split_item)
            if not p.exists():
                raise FileNotFoundError(f"{split} path not found: {p}")

    return d

def build_combined_yaml(base: dict,
                        extra_train: List[str],
                        extra_val: List[str],
                        out_path: Path) -> Path:
    """Append extra train/val sources (dirs or lists) to avoid catastrophic forgetting."""
    combined = dict(base)  # shallow copy
    combined["train"] = _to_list(combined["train"]) + extra_train
    combined["val"]   = _to_list(combined["val"]) + (extra_val if extra_val else extra_train)
    _save_yaml(combined, out_path)
    return out_path

def main():
    ap = argparse.ArgumentParser(description="YOLOv8 transfer-learning to 81 classes (COCO+face) with replay data")
    ap.add_argument("--data", type=Path, required=True, help="Path to your 81-class data.yaml")
    ap.add_argument("--weights", type=str, default="yolov8s.pt",
                    help="Pretrained base (COCO) or your own .pt (s/m/l/x).")
    ap.add_argument("--epochs", type=int, default=1)  # <-- changed to 1 epoch
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="Auto-batch if -1")
    ap.add_argument("--device", type=str, default="0", help="'cpu', 'mps', or CUDA index like '0'")
    ap.add_argument("--project", type=str, default="runs/train")
    ap.add_argument("--name", type=str, default="yolov8_81_replay")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export", type=str, default="", help="onnx, openvino, etc. ('' to skip)")

    # Replay/mixing to preserve 80 classes
    ap.add_argument("--extra-train", action="append", default=[],
                    help="Extra train sources (repeat flag). E.g. /data/old80/images/train")
    ap.add_argument("--extra-val", action="append", default=[],
                    help="Extra val sources (repeat flag). Default: mirrors --extra-train")

    # Transfer-learning stability
    ap.add_argument("--freeze", type=int, default=10,
                    help="Freeze first N layers (0 = no freeze). Helps retain low-level features.")

    # Test folder (not a single image anymore)
    ap.add_argument("--test-image", type=Path, default=Path("./data/demo_folder"),
                    help="Path to a demo folder for a quick post-training prediction.")


    args = ap.parse_args()

    # Validate base YAML (expects 81 classes)
    base = sanity_check_81(args.data)

    # If user provided any replay sources, write a combined yaml next to input
    use_yaml = args.data
    if args.extra_train or args.extra_val:
        out_yaml = args.data.parent / f"combined_{args.data.stem}.yaml"
        use_yaml = build_combined_yaml(base, args.extra_train, args.extra_val, out_yaml)
        print(f"[INFO] Using combined data yaml: {use_yaml}")
        print(f"[INFO] train sources: { _load_yaml(use_yaml)['train'] }")
        print(f"[INFO] val sources:   { _load_yaml(use_yaml)['val'] }")

    # Train
    model = YOLO(args.weights)
    results = model.train(
        data=str(use_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        cos_lr=True,
        close_mosaic=10,
        patience=args.patience,
        amp=True,
        cache="ram",
        freeze=args.freeze
    )

    # Validate best weights
    best = Path(results.save_dir) / "weights" / "best.pt"
    model = YOLO(best)
    model.val(data=str(use_yaml), imgsz=args.imgsz, device=args.device)

    # Optional export
    if args.export:
        model.export(format=args.export)

    # Save class list for reference
    cls_txt = Path(results.save_dir) / "classes.txt"
    cls_txt.write_text("\n".join(base["names"]))
    print(f"\n[OK] Classes saved to: {cls_txt}")

        # ---- Quick test inference on a demo folder ----
    if args.test_image.exists():
        pred_dir = Path(results.save_dir) / "predict"
        print(f"[INFO] Running demo prediction on folder: {args.test_image}")
        _ = model.predict(
            source=str(args.test_image),  # now a folder path
            imgsz=args.imgsz,
            device=args.device,
            conf=0.45,
            iou=0.5,
            max_det=50,
            agnostic_nms=True,
            project=str(results.save_dir),
            name="predict",
            save=True,
            verbose=False
        )
        print(f"[OK] Predictions saved under: {pred_dir}")
    else:
        print(f"[WARN] Demo folder not found at {args.test_image}. Skipping prediction.", file=sys.stderr)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
