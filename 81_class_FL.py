#!/usr/bin/env python3
# 81_class_FL.py
# Federated YOLOv8 (single machine, sequential clients) with class-based design.
# - Auto-build an 81-class seed from an 80-class checkpoint (so COCO logits are preserved).
# - Server orchestrates rounds, aggregation, checkpoints.
# - Client trains locally using its own data.yaml, returns EMA/best or last weights.
# - FedAvg: float tensors -> weighted average; non-float -> first client's value.
# - Row-protection: for class-dimensioned (81) tensors, COCO rows 0..79 are anchored toward the seed early rounds.
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
import os

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

def global_val_on_client(yolo_ckpt: Path, client_yaml: Path, imgsz: int, device: str) -> float:
    """Return mAP50-95 of the global model on a client's val set."""
    y = YOLO(str(yolo_ckpt))
    res = y.val(data=str(client_yaml), imgsz=imgsz, device=device, verbose=False)
    try:
        return float(res.box.map)
    except Exception:
        return 0.0

# -----------------------------
# Seed builder (80 -> 81)
# -----------------------------
def _detect_head_nc(sd: Dict[str, torch.Tensor]) -> int:
    # Heuristic: find a tensor with class-dim (#out channels) used for classification conv/bias.
    # In YOLOv8 detect head, those tensors have shape [nc, ...] (first dim == nc) and small trailing shapes.
    candidates = [t.shape[0] for k, t in sd.items() if t.ndim >= 1 and t.shape[0] in (80, 81)]
    return max(candidates) if candidates else -1

def ensure_81_seed(weights_path: Path, client_yaml_81: Path, out_seed_path: Path,
                   force: bool = False) -> Path:
    """
    Ensure we have an 81-class checkpoint to start FL.
    - If `weights_path` already has nc=81 (detect head), return it.
    - Else, auto-create an 81-class seed by asking Ultralytics to adapt head to nc=81.
      We do a zero/one-epoch "init" run on CPU with big freeze to avoid real training.
    - Save to `out_seed_path` and return that path.
    """
    print(f"[SEED] Checking weights: {weights_path}")
    y0 = YOLO(str(weights_path))
    sd0 = y0.model.state_dict()
    nc0 = _detect_head_nc(sd0)
    if nc0 == 81 and not force:
        print("[SEED] Weights already 81-class; using as-is.")
        return weights_path

    print("[SEED] Expanding to 81 classes using client YAML head config...")
    # Try epochs=0 first (may succeed in reconfiguring head)
    runs_root = Path("runs") / "seed_init"
    runs_root.mkdir(parents=True, exist_ok=True)
    try:
        y1 = YOLO(str(weights_path))
        _ = y1.train(
            data=str(client_yaml_81),
            epochs=0,
            imgsz=640,
            device="cpu",
            project=str(runs_root),
            name="e0",
            verbose=False,
            freeze=999,
            close_mosaic=0,
            cos_lr=False,
            patience=0,
            batch=4
        )
        # After this, try to save a normalized checkpoint
        tmp = out_seed_path.with_suffix(".tmp.pt")
        y1.save(str(tmp))
        ytest = YOLO(str(tmp))
        nc1 = _detect_head_nc(ytest.model.state_dict())
        if nc1 == 81:
            Path(tmp).rename(out_seed_path)
            print(f"[SEED] Wrote 81-class seed (epochs=0 init): {out_seed_path}")
            return out_seed_path
    except Exception as e:
        print(f"[SEED] epochs=0 init failed or not supported ({e}). Falling back to epochs=1.")

    # Fallback: epochs=1 with heavy freeze to just reshape head
    y2 = YOLO(str(weights_path))
    res = y2.train(
        data=str(client_yaml_81),
        epochs=1,
        imgsz=640,
        device="cpu",
        project=str(runs_root),
        name="e1",
        verbose=False,
        freeze=999,
        close_mosaic=0,
        cos_lr=False,
        patience=0,
        batch=4
    )
    # pick last.pt
    last_pt = Path(res.save_dir) / "weights" / "last.pt"
    if not last_pt.exists():
        # fallback to best.pt
        last_pt = Path(res.save_dir) / "weights" / "best.pt"
    if not last_pt.exists():
        raise FileNotFoundError("[SEED] Could not find adapted seed checkpoint after fallback run.")

    shutil.copy(last_pt, out_seed_path)
    print(f"[SEED] Wrote 81-class seed (epochs=1 fallback): {out_seed_path}")
    return out_seed_path

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
                   use_last: bool = True,
                   close_mosaic: int = 0,
                   cos_lr: bool = False) -> Tuple[Dict[str, torch.Tensor], dict, int, Path]:
        """
        Start from global_ckpt, train locally, return:
          - client state_dict (from last/best .pt),
          - metrics dict (map50_95, map50),
          - n_train_images (weight),
          - path to ckpt used for template (last/best)
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
            cos_lr=cos_lr,
            close_mosaic=close_mosaic,
            patience=self.patience,
            amp=True,
            cache="ram",
            freeze=freeze,
            verbose=False,
        )
        ckpt_name = "last.pt" if use_last else "best.pt"
        use_pt = Path(results.save_dir) / "weights" / ckpt_name

        sd = yolo_get_sd_from_pt(use_pt)

        # Optional quick val (collect minimal)
        y_best = YOLO(str(use_pt))
        val_res = y_best.val(data=str(self.data_yaml), imgsz=self.imgsz, device=self.device, verbose=False)
        metrics = {"map50_95": None, "map50": None}
        if hasattr(val_res, "box") and hasattr(val_res.box, "map"):
            try:
                metrics["map50_95"] = float(val_res.box.map)  # mAP@[.5:.95]
                metrics["map50"] = float(val_res.box.maps[0]) if len(val_res.box.maps) > 0 else None
            except Exception:
                pass

        return sd, metrics, self.train_images, use_pt

# -----------------------------
# Federated Server
# -----------------------------
class FederatedServer:
    def __init__(self,
                 clients: List[FederatedClient],
                 init_ckpt: Path,        # seed global weights (should be 81-class)
                 outdir: Path = Path("runs_fl"),
                 rounds: int = 5,
                 freeze0: int = 10,
                 unfreeze_every: int = 5,
                 protect_rows: bool = True,
                 alpha_start: float = 0.90,
                 alpha_mid: float = 0.70,
                 alpha_late: float = 0.50,
                 rounds_mid: int = 3,
                 rounds_late: int = 6):
        self.clients = clients
        self.rounds = rounds
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        # Bootstrap global checkpoint file (must be 81-class now)
        init_ckpt = Path(init_ckpt)
        self.global_ckpt = init_ckpt
        # Freeze schedule
        self.freeze0 = freeze0
        self.unfreeze_every = max(1, unfreeze_every)
        self.history: List[dict] = []

        # Keep a copy of seed head rows for anchoring
        self.protect_rows = protect_rows
        self.alpha_start = alpha_start
        self.alpha_mid = alpha_mid
        self.alpha_late = alpha_late
        self.rounds_mid = rounds_mid
        self.rounds_late = rounds_late

        self.seed_head: Dict[str, torch.Tensor] = {}
        try:
            y_seed = YOLO(str(self.global_ckpt))
            sd_seed = y_seed.model.state_dict()
            for k, t in sd_seed.items():
                if t.ndim >= 1 and t.shape[0] == 81:  # class-dimension tensors in head
                    self.seed_head[k] = t.detach().clone().cpu()
            print(f"[SERVER] Cached seed head tensors for protection: {len(self.seed_head)}")
        except Exception as e:
            print(f"[WARN] Could not cache seed head rows (protection disabled): {e}")
            self.protect_rows = False

    def _alpha_for_round(self, r: int) -> float:
        if r <= self.rounds_mid:
            return self.alpha_start
        if r <= self.rounds_late:
            return self.alpha_mid
        return self.alpha_late

    def _aggregate_fedavg(self,
                          client_sds: List[Dict[str, torch.Tensor]],
                          weights: List[float],
                          round_idx: int) -> Dict[str, torch.Tensor]:
        """
        Float tensors: weighted average.
        Non-floats: take the FIRST client's value.
        For class-dimensioned (81) tensors, optionally protect rows 0..79 by anchoring to seed.
        """
        if not client_sds:
            raise ValueError("No client states provided for aggregation")
        keys = list(client_sds[0].keys())
        for sd in client_sds[1:]:
            if list(sd.keys()) != keys:
                raise ValueError("Mismatched state_dict keys across clients")

        total = float(sum(weights))
        w_norm = [float(x) / total for x in weights]

        alpha = self._alpha_for_round(round_idx)

        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            t0 = client_sds[0][k]
            if not t0.dtype.is_floating_point:
                out[k] = t0.clone()
                continue

            # weighted average (float32 accum)
            acc = None
            for sd, w in zip(client_sds, w_norm):
                t = sd[k].to(dtype=torch.float32)
                acc = t * w if acc is None else acc + t * w
            avg = acc

            if (self.protect_rows and k in self.seed_head and avg.ndim >= 1 and avg.shape[0] == 81):
                seed_t = self.seed_head[k].to(avg.device, dtype=avg.dtype)
                protected = avg.clone()
                # blend rows 0..79 toward seed
                protected[:80] = alpha * seed_t[:80] + (1.0 - alpha) * avg[:80]
                # row 80 learns freely
                out[k] = protected.to(dtype=t0.dtype)
            else:
                out[k] = avg.to(dtype=t0.dtype)
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
            demo_iou: float = 0.50,
            use_last: bool = True,
            cos_lr: bool = False,
            close_mosaic: int = 0):
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
                sd, metrics, n_imgs, use_pt = c.train_once(
                    global_ckpt=self.global_ckpt,
                    workdir=round_dir,
                    freeze=freeze,
                    use_last=use_last,
                    close_mosaic=close_mosaic,
                    cos_lr=cos_lr
                )
                client_sds.append(sd)
                weights.append(n_imgs)
                best_pts.append(use_pt)

                # Print per-client metrics
                m = metrics.get("map50_95")
                print(f"[CLIENT {c.client_id}] local mAP50-95: {m}")
                self.history.append({
                    "round": r,
                    "client": c.client_id,
                    "n_imgs": n_imgs,
                    **metrics
                })

            print("[SERVER] Aggregating (float=weighted mean, non-float=first client)...")
            global_sd = self._aggregate_fedavg(client_sds, weights, round_idx=r)

            next_global = round_dir / "global.pt"
            # Use a client template (already 81-class) to avoid head shape mismatch
            ref_template = best_pts[0]
            save_sd_as_pt(global_sd, ref_template, next_global)
            self.global_ckpt = next_global
            print(f"[SERVER] New global saved to: {self.global_ckpt}")

            # Global validation across client val sets
            maps = []
            for c in self.clients:
                gm = global_val_on_client(self.global_ckpt, c.data_yaml, imgsz, device)
                maps.append(gm)
                print(f"[GLOBAL->client {c.client_id}] mAP50-95 on its val: {gm:.4f}")
            if maps:
                print(f"[GLOBAL] Macro-avg mAP50-95 this round: {sum(maps)/len(maps):.4f}")

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
                    help="Initial global weights (.pt) or a hub name (e.g., yolov8s.pt). Can be 80-class.")
    ap.add_argument("--auto-seed", action="store_true",
                    help="Auto-create an 81-class seed from the --weights using the first client's YAML.")
    ap.add_argument("--seed-output", type=str, default="seed_81.pt",
                    help="Where to save the auto-generated 81-class seed.")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--outdir", type=str, default="runs_fl_classes")
    ap.add_argument("--freeze0", type=int, default=15)
    ap.add_argument("--unfreeze-every", type=int, default=3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    # Demo args
    ap.add_argument("--demo-dir", type=str, default="data/demo_folder",
                    help="Folder of images to run quick predictions on after training.")
    ap.add_argument("--demo-conf", type=float, default=0.45,
                    help="Confidence threshold for demo predictions.")
    ap.add_argument("--demo-iou", type=float, default=0.50,
                    help="IoU threshold for NMS during demo predictions.")
    # Training knobs
    ap.add_argument("--use-last", action="store_true",
                    help="Aggregate raw 'last.pt' instead of EMA 'best.pt'. Recommended early.")
    ap.add_argument("--no-cos-lr", action="store_true",
                    help="Disable cosine LR (recommended for tiny local epochs).")
    ap.add_argument("--no-mosaic", action="store_true",
                    help="Disable mosaic (recommended for tiny local epochs).")
    # Protection knobs
    ap.add_argument("--no-protect", action="store_true",
                    help="Disable COCO row protection during aggregation.")
    ap.add_argument("--protect-alpha-start", type=float, default=0.90)
    ap.add_argument("--protect-alpha-mid", type=float, default=0.70)
    ap.add_argument("--protect-alpha-late", type=float, default=0.50)
    ap.add_argument("--protect-rounds-mid", type=int, default=3)
    ap.add_argument("--protect-rounds-late", type=int, default=6)
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
    if len(clients) == 0:
        raise ValueError("No clients found.")

    # Ensure 81-class seed if requested or if weights look 80-class
    weights_path = Path(args.weights)
    if args.auto_seed:
        seed_out = Path(args.outdir) / args.seed_output
        seed_out.parent.mkdir(parents=True, exist_ok=True)
        weights_path = ensure_81_seed(weights_path, clients[0].data_yaml, seed_out, force=False)
    else:
        # Best effort: if weights look 80-class, still try to expand (safer default)
        try:
            ytmp = YOLO(str(weights_path))
            nc = _detect_head_nc(ytmp.model.state_dict())
            if nc == 80:
                print("[INFO] --auto-seed not set, but weights appear to be 80-class. "
                      "It's recommended to pass --auto-seed.")
        except Exception:
            pass

    server = FederatedServer(
        clients=clients,
        init_ckpt=weights_path,
        outdir=Path(args.outdir),
        rounds=args.rounds,
        freeze0=args.freeze0,
        unfreeze_every=args.unfreeze_every,
        protect_rows=(not args.no_protect),
        alpha_start=args.protect_alpha_start,
        alpha_mid=args.protect_alpha_mid,
        alpha_late=args.protect_alpha_late,
        rounds_mid=args.protect_rounds_mid,
        rounds_late=args.protect_rounds_late
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
        use_last=args.use_last,
        cos_lr=(not args.no_cos_lr),
        close_mosaic=(0 if args.no_mosaic else 10),
    )

if __name__ == "__main__":
    main()
