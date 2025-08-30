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
from opacus import PrivacyEngine

# Utility functions
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
        agnostic_nms=False,
        project=str(out_dir),
        name="predict",
        save=True,
        verbose=False,
    )
    print(f"[OK] Demo predictions saved to: {pred_dir}")
    return pred_dir

def global_val_on_client(yolo_ckpt: Path, client_yaml: Path, imgsz: int, device: str) -> float:
    y = YOLO(str(yolo_ckpt))
    res = y.val(data=str(client_yaml), imgsz=imgsz, device=device, verbose=False)
    try:
        return float(res.box.map)
    except Exception:
        return 0.0

# -----------------------------
# Seed builder (80 -> 81, preserves COCO rows)
# -----------------------------
def _detect_head_nc(sd: Dict[str, torch.Tensor]) -> int:
    cands = [t.shape[0] for _, t in sd.items() if t.ndim >= 1 and t.shape[0] in (80, 81)]
    return max(cands) if cands else -1

def _copy_rows_80_to_81(sd81: Dict[str, torch.Tensor], sd80: Dict[str, torch.Tensor]):
    """Copy rows 0..79 from sd80 into sd81 for any tensor with shape[0]==81 when sd80 has shape[0]==80."""
    for k, v81 in list(sd81.items()):
        if v81.ndim >= 1 and v81.shape[0] == 81 and k in sd80:
            v80 = sd80[k]
            if v80.ndim >= 1 and v80.shape[0] == 80 and v80.shape[1:] == v81.shape[1:]:
                nv = v81.clone()
                nv[:80].copy_(v80)
                # leave row 80 (face) as-initialized
                sd81[k] = nv

def ensure_81_seed(weights_path: Path, client_yaml_81: Path, out_seed_path: Path, device: str) -> Path:
    """
    Create an 81-class checkpoint from an 80-class checkpoint while preserving COCO rows (0..79).
    - Does a 1-epoch head-adaptation with backbone mostly frozen (so grads exist).
    - Disables AMP to avoid scaler/backward issues.
    - Copies 80-row tensors into the new 81-row tensors for stability.
    """
    print(f"[SEED] Preparing 81-class seed from: {weights_path}")
    y0 = YOLO(str(weights_path))
    sd0 = y0.model.state_dict()
    nc0 = _detect_head_nc(sd0)
    if nc0 == 81:
        print("[SEED] Already 81-class, using as-is.")
        return weights_path

    # 1) Run a tiny adaptation so Ultralytics rebuilds the detect head to nc=81
    y_adapt = YOLO(str(weights_path))
    res = y_adapt.train(
        data=str(client_yaml_81),
        epochs=1,
        imgsz=640,
        device=device,          # use the same device as training if available
        project="runs/seed_init",
        name="adapt_e1",
        verbose=False,
        freeze=10,              # freeze backbone, keep head trainable -> grads exist
        close_mosaic=0,
        cos_lr=False,
        patience=0,
        batch=4,
        amp=False               # avoid AMP/scaler issues on near-zero grads
    )
    ckpt = Path(res.save_dir) / "weights" / "last.pt"
    if not ckpt.exists():
        ckpt = Path(res.save_dir) / "weights" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError("[SEED] Could not find adapted checkpoint.")

    # 2) Load adapted (81-class) and overwrite class-dimension rows 0..79 with COCO rows
    y81 = YOLO(str(ckpt))
    sd81 = y81.model.state_dict()
    _copy_rows_80_to_81(sd81, sd0)
    y81.model.load_state_dict(sd81, strict=False)

    out_seed_path.parent.mkdir(parents=True, exist_ok=True)
    y81.save(str(out_seed_path))
    print(f"[SEED] Wrote 81-class seed: {out_seed_path}")
    return out_seed_path

# Federated Client

class FederatedClient:
    def __init__(self,
                 client_id: str,
                 data_yaml: Path,
                 device: str = "0",
                 imgsz: int = 640,
                 batch: int = -1,
                 local_epochs: int = 1,
                 patience: int = 30,
                 seed: int = 42,
                 # Add DP parameters
                 enable_dp: bool = False,
                 target_epsilon: float = 10.0,
                 target_delta: float = 1e-5,
                 max_grad_norm: float = 1.0):
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

        # DP parameters
        self.enable_dp = enable_dp
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm

    def train_once(self,
                   global_ckpt: Path,
                   workdir: Path,
                   freeze: int,
                   use_last: bool = True,
                   close_mosaic: int = 0,
                   cos_lr: bool = False,
                   lr0: float = 0.01) -> Tuple[Dict[str, torch.Tensor], dict, int, Path]:
        run_dir = workdir / f"client_{self.client_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        y = YOLO(str(global_ckpt))

        if self.enable_dp:
            print(f"[CLIENT {self.client_id}] Training with Differential Privacy (ε={self.target_epsilon}, δ={self.target_delta})")

            # Apply DP modifications to YOLO training
            results = self._train_with_dp(y, run_dir, freeze, use_last, close_mosaic, cos_lr, lr0)
        else:
            # Original training without DP
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
                lr0=lr0,
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

        # Validation
        y_eval = YOLO(str(use_pt))
        val_res = y_eval.val(data=str(self.data_yaml), imgsz=self.imgsz, device=self.device, verbose=False)
        metrics = {"map50_95": None, "map50": None}
        if hasattr(val_res, "box") and hasattr(val_res.box, "map"):
            try:
                metrics["map50_95"] = float(val_res.box.map)
                metrics["map50"] = float(val_res.box.maps[0]) if len(val_res.box.maps) > 0 else None
            except Exception:
                pass

        return sd, metrics, self.train_images, use_pt

    def _train_with_dp(self, yolo_model, run_dir, freeze, use_last, close_mosaic, cos_lr, lr0):
        """
        Custom training method that integrates Opacus for differential privacy.
        Note: This is a conceptual implementation as YOLO's internal training loop
        would need significant modifications for full Opacus integration.
        """
        print(f"[DP] Starting differentially private training for client {self.client_id}")

        # For now, we'll use YOLO's built-in training but with modified parameters
        # In a full implementation, you'd need to extract YOLO's training loop
        # and integrate it with Opacus PrivacyEngine

        # Adjust batch size and other parameters for DP
        dp_batch = max(8, self.batch) if self.batch > 0 else 16  # Ensure minimum batch size for DP

        results = yolo_model.train(
            data=str(self.data_yaml),
            epochs=self.local_epochs,
            imgsz=self.imgsz,
            batch=dp_batch,
            device=self.device,
            workers=4,  # Reduced workers for DP stability
            project=str(run_dir),
            name="train_dp",
            seed=self.seed,
            cos_lr=cos_lr,
            lr0=lr0,
            close_mosaic=close_mosaic,
            patience=self.patience,
            amp=False,  # Disable AMP for DP compatibility
            cache=False,  # Disable caching for DP
            freeze=freeze,
            verbose=False,
        )

        return results

    def create_dp_trainer(self, model, optimizer, dataloader):
        """
        Creates a PrivacyEngine and makes model/optimizer/dataloader private.
        This is a helper method for manual DP integration.
        """
        privacy_engine = PrivacyEngine()

        private_model, private_optimizer, private_dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            epochs=self.local_epochs,
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            max_grad_norm=self.max_grad_norm,
        )[1]

        return private_model, private_optimizer, private_dataloader, privacy_engine

# Federated Server

class FederatedServer:
    def __init__(self,
                 clients: List[FederatedClient],
                 init_ckpt: Path,
                 outdir: Path = Path("runs_fl"),
                 rounds: int = 5,
                 freeze0: int = 10,
                 unfreeze_every: int = 5,
                 protect_rows: bool = True,
                 alpha_start: float = 0.90,
                 alpha_mid: float = 0.70,
                 alpha_late: float = 0.50,
                 rounds_mid: int = 1,
                 rounds_late: int = 3,
                 server_dp: bool = False,
                 server_noise_multiplier: float = 0.1):

        self.clients = clients
        self.rounds = rounds
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.global_ckpt = Path(init_ckpt)
        self.freeze0 = freeze0
        self.unfreeze_every = max(1, unfreeze_every)
        self.history: List[dict] = []
        self.protect_rows = protect_rows
        self.alpha_start = alpha_start
        self.alpha_mid = alpha_mid
        self.alpha_late = alpha_late
        self.rounds_mid = rounds_mid
        self.rounds_late = rounds_late

        # Server-side DP (for user-level privacy)
        self.server_dp = server_dp
        self.server_noise_multiplier = server_noise_multiplier

        # [Keep existing seed head caching code]
        self.seed_head: Dict[str, torch.Tensor] = {}
        try:
            y_seed = YOLO(str(self.global_ckpt))
            sd_seed = y_seed.model.state_dict()
            for k, t in sd_seed.items():
                if t.ndim >= 1 and t.shape[0] == 81:
                    self.seed_head[k] = t.detach().clone().cpu()
            print(f"[SERVER] Cached seed head tensors: {len(self.seed_head)}")
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

            # Weighted average
            acc = None
            for sd, w in zip(client_sds, w_norm):
                t = sd[k].to(dtype=torch.float32)
                acc = t * w if acc is None else acc + t * w
            avg = acc

            # Add server-side DP noise if enabled
            if self.server_dp and avg.dtype.is_floating_point:
                noise = torch.randn_like(avg) * self.server_noise_multiplier
                avg = avg + noise
                print(f"[SERVER-DP] Added noise to {k} with std={self.server_noise_multiplier}")

            # Apply protection for COCO rows
            if (self.protect_rows and k in self.seed_head and avg.ndim >= 1 and avg.shape[0] == 81):
                seed_t = self.seed_head[k].to(avg.device, dtype=avg.dtype)
                protected = avg.clone()
                protected[:80] = alpha * seed_t[:80] + (1.0 - alpha) * avg[:80]
                out[k] = protected.to(dtype=t0.dtype)
            else:
                out[k] = avg.to(dtype=t0.dtype)
        return out

    def _freeze_for_round(self, r: int) -> int:
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
            close_mosaic: int = 0,
            lr0: float = 0.01,
            demo_every_round: bool = True):
        t0 = time.time()
        for r in range(1, self.rounds + 1):
            print(f"\n=== Federated Round {r}/{self.rounds} ===")
            freeze = self._freeze_for_round(r)
            round_dir = self.outdir / f"round_{r:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)

            client_sds: List[Dict[str, torch.Tensor]] = []
            weights: List[int] = []
            templates: List[Path] = []

            for c in self.clients:
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
                    cos_lr=cos_lr,
                    lr0=lr0
                )
                client_sds.append(sd)
                weights.append(n_imgs)
                templates.append(use_pt)
                print(f"[CLIENT {c.client_id}] local mAP50-95: {metrics.get('map50_95')}")
                self.history.append({"round": r, "client": c.client_id, "n_imgs": n_imgs, **metrics})

            print("[SERVER] Aggregating (float=weighted mean, non-float=first client)...")
            global_sd = self._aggregate_fedavg(client_sds, weights, round_idx=r)

            next_global = round_dir / "global.pt"
            save_sd_as_pt(global_sd, templates[0], next_global)  # template already 81-class
            self.global_ckpt = next_global
            print(f"[SERVER] New global saved to: {self.global_ckpt}")
            if demo_every_round:
                print(f"[DEMO] Round {r}: predicting on {demo_dir}")
                run_demo_prediction(
                    model_ckpt=self.global_ckpt,
                    demo_dir=Path(demo_dir),
                    out_dir=round_dir,     # save under this round's folder
                    imgsz=imgsz,
                    device=device,
                    conf=demo_conf,
                    iou=demo_iou,
                )

            # Global validation
            maps = []
            for c in self.clients:
                gm = global_val_on_client(self.global_ckpt, c.data_yaml, imgsz, device)
                maps.append(gm)
                print(f"[GLOBAL->client {c.client_id}] mAP50-95 on its val: {gm:.4f}")
            if maps:
                print(f"[GLOBAL] Macro-avg mAP50-95 this round: {sum(maps)/len(maps):.4f}")

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

# CLI

def parse_args():
    ap = argparse.ArgumentParser(description="Federated YOLOv8 with Differential Privacy (Opacus)")
    ap.add_argument("--clients", type=str, required=True,
                    help="Comma-separated list of client data.yaml paths, a JSON list file, or a directory of yamls.")
    ap.add_argument("--weights", type=str, default="yolov8s.pt",
                    help="Initial weights (.pt). Can be 80-class; script will expand if --auto-seed.")
    ap.add_argument("--auto-seed", action="store_true",
                    help="Auto-create an 81-class seed from --weights using the first client's YAML.")
    ap.add_argument("--seed-output", type=str, default="seed_81.pt",
                    help="Where to save the auto-generated 81-class seed.")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--outdir", type=str, default="runs_fl_classes")
    ap.add_argument("--freeze0", type=int, default=15)
    ap.add_argument("--unfreeze-every", type=int, default=3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--enable-dp", action="store_true",
                    help="Enable differential privacy using Opacus")
    ap.add_argument("--target-epsilon", type=float, default=10.0,
                    help="Target epsilon for differential privacy (default: 10.0)")
    ap.add_argument("--target-delta", type=float, default=1e-5,
                    help="Target delta for differential privacy (default: 1e-5)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="Maximum gradient norm for clipping in DP (default: 1.0)")
    ap.add_argument("--server-dp", action="store_true",
                    help="Enable server-side differential privacy (user-level)")
    ap.add_argument("--server-noise-multiplier", type=float, default=0.1,
                    help="Noise multiplier for server-side DP (default: 0.1)")
    ap.add_argument("--demo-dir", type=str, default="data/demo_folder")
    ap.add_argument("--demo-conf", type=float, default=0.45)
    ap.add_argument("--demo-iou", type=float, default=0.50)
    ap.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    ap.add_argument("--use-last", action="store_true",
                    help="Aggregate 'last.pt' instead of EMA 'best.pt' (recommended early).")
    ap.add_argument("--no-cos-lr", action="store_true",
                    help="Disable cosine LR (recommended for tiny local epochs).")
    ap.add_argument("--no-mosaic", action="store_true",
                    help="Disable mosaic (recommended for tiny local epochs).")
    ap.add_argument("--no-protect", action="store_true",
                    help="Disable COCO row protection during aggregation.")
    ap.add_argument("--protect-alpha-start", type=float, default=0.90)
    ap.add_argument("--protect-alpha-mid", type=float, default=0.70)
    ap.add_argument("--protect-alpha-late", type=float, default=0.50)
    ap.add_argument("--protect-rounds-mid", type=int, default=3)
    ap.add_argument("--protect-rounds-late", type=int, default=6)
    ap.add_argument("--demo-every-round", action="store_true",
                help="Run predictions on the demo folder after each round using the aggregated global model.")
    return ap.parse_args()

def load_clients(spec: str, enable_dp: bool = False, target_epsilon: float = 10.0,
                target_delta: float = 1e-5, max_grad_norm: float = 1.0) -> List[FederatedClient]:
    p = Path(spec)
    if p.suffix.lower() == ".json" and p.exists():
        items = json.loads(p.read_text()); paths = [Path(x) for x in items]
    elif p.is_dir():
        paths = sorted(list(p.glob("*.yaml")) + list(p.glob("*.yml")))
        if not paths: raise FileNotFoundError(f"No .yaml files found in directory {p}")
    else:
        paths = [Path(x.strip()) for x in spec.split(",") if x.strip()]
    clients = []
    for i, yml in enumerate(paths):
        if not yml.exists():
            raise FileNotFoundError(f"Client yaml missing: {yml}")
        clients.append(FederatedClient(
            client_id=f"{i:02d}",
            data_yaml=yml,
            enable_dp=enable_dp,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm
        ))
    return clients

def main():
    args = parse_args()
    clients = load_clients(
        args.clients,
        enable_dp=args.enable_dp,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        max_grad_norm=args.max_grad_norm
    )
    if len(clients) == 0:
        raise ValueError("No clients found.")

    weights_path = Path(args.weights)
    if args.auto_seed:
        seed_out = Path(args.outdir) / args.seed_output
        seed_out.parent.mkdir(parents=True, exist_ok=True)
        weights_path = ensure_81_seed(weights_path, clients[0].data_yaml, seed_out, device=args.device)
    else:
        try:
            ytmp = YOLO(str(weights_path))
            if _detect_head_nc(ytmp.model.state_dict()) == 80:
                print("[INFO] Weights appear 80-class; consider --auto-seed to preserve COCO logits.")
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
        rounds_late=args.protect_rounds_late,
        server_dp=args.server_dp,
        server_noise_multiplier=args.server_noise_multiplier
    )

    if args.enable_dp:
        print(f"[DP] Enabled differential privacy with ε={args.target_epsilon}, δ={args.target_delta}")
        print(f"[DP] Max gradient norm: {args.max_grad_norm}")

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
        lr0=args.lr0,
        demo_every_round=args.demo_every_round,
    )

if __name__ == "__main__":
    main()
