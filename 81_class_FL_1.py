#!/usr/bin/env python3
# 81_class_FL.py
# Federated YOLOv8 (single machine, sequential clients) with class-based design.
# - Server orchestrates rounds, aggregation, checkpoints.
# - Client trains locally using its own data.yaml, returns EMA weights.
# - FedAvg: float tensors -> weighted average; non-float -> take first client's value.
# - After training, run a demo prediction on a folder (default: data/demo_folder).

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
from ultralytics import YOLO
import glob

# -----------------------------
# Utilities
# -----------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def read_yaml(p: Path) -> dict:
    with p.open() as f:
        return yaml.safe_load(f)

def count_images_from_yaml(data_yaml: Path) -> int:
    d = read_yaml(data_yaml)
    root = Path(d.get("path", ""))  # may be empty
    def _count_one(item: str) -> int:
        p = Path(item)
        if not p.is_absolute():
            p = root / p
        if p.is_dir():
            c = 0
            for ext in IMG_EXTS:
                c += len(list(p.rglob(f"*{ext}")))
            return c
        if p.is_file() and p.suffix.lower() == ".txt":
            try:
                return sum(1 for _ in p.open())
            except Exception:
                return 0
        if p.exists() and any(str(p).lower().endswith(ext) for ext in IMG_EXTS):
            return 1
        # try glob as a fallback
        c = 0
        for ext in IMG_EXTS:
            c += len(glob.glob(str(p) + f"/**/*{ext}", recursive=True))
        return c

    train_field = d["train"]
    if isinstance(train_field, list):
        return sum(_count_one(it) for it in train_field)
    return _count_one(train_field)

def check_81_classes(data_yaml: Path):
    d = read_yaml(data_yaml)
    for k in ("names", "nc", "train", "val"):
        if k not in d:
            raise ValueError(f"Missing key '{k}' in {data_yaml}")
    names = d["names"]
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if len(names) != 81 or int(d["nc"]) != 81:
        raise ValueError(f"{data_yaml}: expected 81 classes, got nc={d['nc']} len(names)={len(names)}")
    if names[80] != "face":
        print(f"[WARN] {data_yaml}: names[80] != 'face' (got '{names[80]}'). Verify class indexing.")

def yolo_get_sd_from_pt(pt_path: Path) -> Dict[str, torch.Tensor]:
    y = YOLO(str(pt_path))
    return {k: v.detach().clone().cpu() for k, v in y.model.state_dict().items()}

def save_sd_as_pt(state_dict: Dict[str, torch.Tensor], ref_weights: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ref_weights, out_path)
    y = YOLO(str(out_path))
    y.model.load_state_dict(state_dict, strict=True)
    y.save(str(out_path))  # normalize checkpoint structure

def run_demo_prediction(model_ckpt: Path, demo_dir: Path, out_dir: Path,
                        imgsz: int, device: str, conf: float, iou: float) -> Path:
    """
    Run YOLO predictions on a folder; saves under out_dir/'predict'.
    Returns the predict directory path.
    """
    if not demo_dir.exists():
        print(f"[WARN] Demo folder not found: {demo_dir}. Skipping demo.")
        return out_dir / "predict"
    y = YOLO(str(model_ckpt))
    pred_dir = out_dir / "predict"
    _ = y.predict(
        source=str(demo_dir),
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        max_det=50,
        agnostic_nms=True,
        project=str(out_dir),
        name="predict",
        save=True,
        verbose=False,
    )
    print(f"[OK] Demo predictions saved to: {pred_dir}")
    return pred_dir

# -----------------------------
# Federated Client
# -----------------------------
class FederatedClient:
    def __init__(self,
                 client_id: str,
                 data_yaml: Path,
                 device: str = "0",
                 imgsz: int = 640,
                 batch: int = -1,
                 local_epochs: int = 1,
                 patience: int = 30,
                 seed: int = 42):
        self.client_id = client_id
        self.data_yaml = Path(data_yaml)
        self.device = device
        self.imgsz = imgsz
        self.batch = batch
        self.local_epochs = local_epochs
        self.patience = patience
        self.seed = seed
        self.train_images = count_images_from_yaml(self.data_yaml)

        check_81_classes(self.data_yaml)

    def train_once(self,
                   global_ckpt: Path,
                   workdir: Path,
                   freeze: int,
                   close_mosaic: int = 10) -> Tuple[Dict[str, torch.Tensor], dict, int, Path]:
        """
        Start from global_ckpt, train locally, return:
          - client EMA state_dict (from best.pt),
          - metrics dict (minimal),
          - n_train_images (weight),
          - path to best.pt
        """
        run_dir = workdir / f"client_{self.client_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        y = YOLO(str(global_ckpt))
        results = y.train(
            data=str(self.data_yaml),
            epochs=self.local_epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            device=self.device,
            workers=8,
            project=str(run_dir),
            name="train",
            seed=self.seed,
            cos_lr=True,
            close_mosaic=close_mosaic,
            patience=self.patience,
            amp=True,
            cache="ram",
            freeze=freeze,
            verbose=False,
        )
        best_pt = Path(results.save_dir) / "weights" / "best.pt"

        # Best is EMA by default; pull model weights
        sd = yolo_get_sd_from_pt(best_pt)

        # Optional quick val (collect minimal)
        y_best = YOLO(str(best_pt))
        val_res = y_best.val(data=str(self.data_yaml), imgsz=self.imgsz, device=self.device, verbose=False)
        metrics = {"map50_95": None, "map50": None}
        if hasattr(val_res, "box") and hasattr(val_res.box, "map"):
            try:
                metrics["map50_95"] = float(val_res.box.map)  # mAP@[.5:.95]
                metrics["map50"] = float(val_res.box.maps[0]) if len(val_res.box.maps) > 0 else None
            except Exception:
                pass

        return sd, metrics, self.train_images, best_pt

# -----------------------------
# Federated Server
# -----------------------------
class FederatedServer:
    def __init__(self,
                 clients: List[FederatedClient],
                 init_ckpt: Path,        # seed global weights (e.g., yolov8s.pt or prior 81-class pt)
                 outdir: Path = Path("runs_fl"),
                 rounds: int = 5,
                 freeze0: int = 10,
                 unfreeze_every: int = 5):
        self.clients = clients
        self.rounds = rounds
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        # Bootstrap global checkpoint file
        init_ckpt = Path(init_ckpt)
        if not init_ckpt.exists():
            # Allow a name like 'yolov8s.pt' (downloadable by Ultralytics)
            y0 = YOLO(str(init_ckpt))
            tmp = self.outdir / "global_round_000.pt"
            y0.save(str(tmp))
            self.global_ckpt = tmp
        else:
            self.global_ckpt = init_ckpt

        self.freeze0 = freeze0
        self.unfreeze_every = max(1, unfreeze_every)
        self.history: List[dict] = []

    @staticmethod
    def _aggregate_fedavg(client_sds: List[Dict[str, torch.Tensor]],
                          weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        Float tensors: weighted average.
        Non-floats (int, bool, quantized, etc.): take the FIRST client's value.
        """
        if not client_sds:
            raise ValueError("No client states provided for aggregation")
        keys = list(client_sds[0].keys())

        # Sanity: intersect keys across all clients (strictness helps)
        for sd in client_sds[1:]:
            if list(sd.keys()) != keys:
                raise ValueError("Mismatched state_dict keys across clients")

        total = float(sum(weights))
        w_norm = [float(x) / total for x in weights]

        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            t0 = client_sds[0][k]
            if t0.dtype.is_floating_point:
                acc = None
                for sd, w in zip(client_sds, w_norm):
                    t = sd[k].to(dtype=torch.float32)
                    acc = t * w if acc is None else acc + t * w
                out[k] = acc.to(dtype=t0.dtype)
            else:
                # Special case: copy first client's value for non-float
                out[k] = t0.clone()
        return out

    def _freeze_for_round(self, r: int) -> int:
        # Unfreeze 5 layers every `unfreeze_every` rounds
        return max(0, self.freeze0 - 5 * ((r - 1) // self.unfreeze_every))

    def run(self,
            imgsz: int = 640,
            batch: int = -1,
            device: str = "0",
            local_epochs: int = 1,
            patience: int = 30,
            seed: int = 42,
            demo_dir: str = "data/demo_folder",
            demo_conf: float = 0.45,
            demo_iou: float = 0.50):
        """
        Orchestrates FL rounds. Each client inherits these defaults unless it set its own.
        After all rounds, runs demo predictions with final global on demo_dir.
        """
        t0 = time.time()
        for r in range(1, self.rounds + 1):
            print(f"\n=== Federated Round {r}/{self.rounds} ===")
            freeze = self._freeze_for_round(r)
            round_dir = self.outdir / f"round_{r:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)

            client_sds: List[Dict[str, torch.Tensor]] = []
            weights: List[int] = []
            best_pts: List[Path] = []

            for c in self.clients:
                # allow server to override global defaults on the fly
                c.imgsz = imgsz
                c.batch = batch
                c.device = device
                c.local_epochs = local_epochs
                c.patience = patience
                c.seed = seed

                print(f"[CLIENT {c.client_id}] train_images≈{c.train_images}, freeze={freeze}")
                sd, metrics, n_imgs, best_pt = c.train_once(
                    global_ckpt=self.global_ckpt,
                    workdir=round_dir,
                    freeze=freeze
                )
                client_sds.append(sd)
                weights.append(n_imgs)
                best_pts.append(best_pt)
                self.history.append({
                    "round": r,
                    "client": c.client_id,
                    "n_imgs": n_imgs,
                    **metrics
                })

            print("[SERVER] Aggregating (FedAvg: float=weighted mean, non-float=first client)...")
            global_sd = self._aggregate_fedavg(client_sds, weights)

            next_global = round_dir / "global.pt"
            # Use a client template (already 81-class) to avoid head shape mismatch
            ref_template = best_pts[0]
            save_sd_as_pt(global_sd, ref_template, next_global)
            self.global_ckpt = next_global
            print(f"[SERVER] New global saved to: {self.global_ckpt}")

        # -------------------------
        # After training: run demo
        # -------------------------
        final_dir = self.outdir / f"round_{self.rounds:03d}"
        print(f"\n[DEMO] Running predictions on: {demo_dir}")
        run_demo_prediction(
            model_ckpt=self.global_ckpt,
            demo_dir=Path(demo_dir),
            out_dir=final_dir,
            imgsz=imgsz,
            device=device,
            conf=demo_conf,
            iou=demo_iou,
        )

        dur = time.time() - t0
        hist_path = self.outdir / "metrics_history.json"
        with hist_path.open("w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n[OK] Finished {self.rounds} rounds in {dur/60:.1f} min. Metrics -> {hist_path}")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Federated YOLOv8 (class-based, single machine)")
    ap.add_argument("--clients", type=str, required=True,
                    help="Comma-separated list of client data.yaml paths, a JSON list file, or a directory of yamls.")
    ap.add_argument("--weights", type=str, default="yolov8s.pt",
                    help="Initial global weights (.pt) or a hub name (e.g., yolov8s.pt).")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--outdir", type=str, default="runs_fl_classes")
    ap.add_argument("--freeze0", type=int, default=10)
    ap.add_argument("--unfreeze-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    # Demo args
    ap.add_argument("--demo-dir", type=str, default="data/demo_folder",
                    help="Folder of images to run quick predictions on after training.")
    ap.add_argument("--demo-conf", type=float, default=0.45,
                    help="Confidence threshold for demo predictions.")
    ap.add_argument("--demo-iou", type=float, default=0.50,
                    help="IoU threshold for NMS during demo predictions.")
    return ap.parse_args()

def load_clients(spec: str) -> List[FederatedClient]:
    p = Path(spec)

    # Case 1: JSON file
    if p.suffix.lower() == ".json" and p.exists():
        items = json.loads(p.read_text())
        paths = [Path(x) for x in items]

    # Case 2: Directory containing YAML files
    elif p.is_dir():
        paths = sorted(list(p.glob("*.yaml")) + list(p.glob("*.yml")))
        if not paths:
            raise FileNotFoundError(f"No .yaml files found in directory {p}")

    # Case 3: Comma-separated string
    else:
        paths = [Path(x.strip()) for x in spec.split(",") if x.strip()]

    clients = []
    for i, yml in enumerate(paths):
        if not yml.exists():
            raise FileNotFoundError(f"Client yaml missing: {yml}")
        clients.append(FederatedClient(client_id=f"{i:02d}", data_yaml=yml))
    return clients


def main():
    args = parse_args()
    clients = load_clients(args.clients)

    server = FederatedServer(
        clients=clients,
        init_ckpt=Path(args.weights),
        outdir=Path(args.outdir),
        rounds=args.rounds,
        freeze0=args.freeze0,
        unfreeze_every=args.unfreeze_every
    )
    server.run(
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        local_epochs=args.local_epochs,
        patience=args.patience,
        seed=args.seed,
        demo_dir=args.demo_dir,
        demo_conf=args.demo_conf,
        demo_iou=args.demo_iou,
    )

if __name__ == "__main__":
    main()
