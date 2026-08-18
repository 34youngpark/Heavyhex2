#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from model.data import npz_path
from dataset_generation.heavyhex33_stim import detectors_from_tensor
from baseline.mwpm import build_matching

DISTANCE = 3
BLIND = "realistic/dp0.001_mf0.01_rf0.01_gd0.008_id0.0"

def noise_tag(noise_str):
    """noise string에서 마지막 부분 추출 (id0.0, id0.002 등)"""
    parts = noise_str.split("_")
    for p in parts:
        if p.startswith("id"):
            return p
    return "unknown"

def ckpt_path(ckpt_dir, noise, cycles, p, seed="s1"):
    """전체 noise string을 그대로 사용"""
    tag = f"d{DISTANCE}_c{cycles}_p{p}_{noise}_{seed}"
    return Path(ckpt_dir) / f"CNN_{tag}.pt"

def load_model(path, cycles, device):
    from model.cnn_skeleton import HeavyHexCNN
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = HeavyHexCNN(in_channels=2 * cycles)
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device).eval()

def cnn_predict(model, features, device, batch=8192):
    out = []
    with torch.no_grad():
        for i in range(0, features.shape[0], batch):
            xb = torch.from_numpy(np.ascontiguousarray(features[i:i + batch])).to(device)
            _, logits = model(xb)
            out.append(logits.float().cpu().numpy().ravel())
    return np.concatenate(out)

def boot_ci(wrong_a, wrong_b, n_boot, rng):
    if n_boot <= 0:
        return None, None
    n = wrong_a.shape[0]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = wrong_a[idx].mean() - wrong_b[idx].mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--rate", type=float, default=0.005)
    parser.add_argument("-e", "--error-type", default="X")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--ckpt-dir", default="checkpoint")
    parser.add_argument("--outdir", default="results/cross")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-mwpm", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"cross_p{args.rate}_c{args.cycles}.csv"
    
    print(f"[Cross-evaluation]")
    print(f"  device: {device}")
    
    test_noise = "realistic/dp0.001_mf0.01_rf0.01_gd0.008_id0.002"
    test_path = npz_path(args.data_dir, test_noise, "test", args.cycles, args.rate, args.error_type)
    
    if not Path(test_path).exists():
        print(f"ERROR: test dataset not found: {test_path}")
        sys.exit(1)
    
    data = np.load(test_path)
    features = data["features"]
    labels = data["labels"]
    y = data["logical_labels"]
    
    n_shots = y.shape[0]
    raw_ler = y.mean()
    print(f"  test shots: {n_shots:,}")
    print(f"  raw LER: {raw_ler:.4f}")
    
    rows = []
    wrong = {}
    
    print("\n[CNN models]")
    for train_noise, label in [
        ("realistic/dp0.001_mf0.01_rf0.01_gd0.008_id0.0", "CNN[train=id0.0]"),
        ("realistic/dp0.001_mf0.01_rf0.01_gd0.008_id0.002", "CNN[train=id0.002]"),
    ]:
        cp = ckpt_path(args.ckpt_dir, train_noise, args.cycles, args.rate, "s1")
        print(f"  loading: {cp}")
        if not cp.exists():
            print(f"    NOT FOUND")
            continue
        
        model = load_model(cp, args.cycles, device)
        logits = cnn_predict(model, features, device)
        pred = (logits > 0).astype(np.uint8)
        
        w = (pred != y).astype(np.float64)
        ler = w.mean()
        wrong[label] = w
        
        rows.append({
            "test_noise": noise_tag(test_noise),
            "decoder": label,
            "train_noise": noise_tag(train_noise),
            "matched": 1 if train_noise == test_noise else 0,
            "ler": ler,
            "raw_ler": raw_ler,
            "shots": n_shots,
        })
        print(f"    {label:<30} LER={ler:.4f}")
    
    if not args.no_mwpm:
        print("\n[MWPM models]")
        det = detectors_from_tensor(features, labels)
        
        for train_noise, label in [
            ("realistic/dp0.001_mf0.01_rf0.01_gd0.008_id0.0", "MWPM[blind]"),
            ("realistic/dp0.001_mf0.01_rf0.01_gd0.008_id0.002", "MWPM[aware]"),
        ]:
            try:
                matching = build_matching(args.cycles, args.error_type, args.rate, train_noise)
                pred = np.asarray(matching.decode_batch(det), dtype=np.uint8)
                pred = pred.reshape(pred.shape[0], -1)[:, 0]
                
                w = (pred != y).astype(np.float64)
                ler = w.mean()
                wrong[label] = w
                
                rows.append({
                    "test_noise": noise_tag(test_noise),
                    "decoder": label,
                    "train_noise": noise_tag(train_noise),
                    "matched": 1 if train_noise == test_noise else 0,
                    "ler": ler,
                    "raw_ler": raw_ler,
                    "shots": n_shots,
                })
                print(f"  {label:<30} LER={ler:.4f}")
            except Exception as e:
                print(f"  SKIP {label}: {e}")
    
    print("\n[Analysis]")
    
    if "CNN[train=id0.0]" in wrong and "CNN[train=id0.002]" in wrong:
        cnn_blind_w = wrong["CNN[train=id0.0]"]
        cnn_aware_w = wrong["CNN[train=id0.002]"]
        
        delta_cnn = cnn_blind_w.mean() - cnn_aware_w.mean()
        lo, hi = boot_ci(cnn_blind_w, cnn_aware_w, args.bootstrap, rng)
        
        print(f"  Δ_CNN (blind - aware) = {delta_cnn:+.5f}")
        if lo is not None:
            print(f"    95% CI: [{lo:+.5f}, {hi:+.5f}]")
    
    if "MWPM[blind]" in wrong and "MWPM[aware]" in wrong:
        mwpm_blind_w = wrong["MWPM[blind]"]
        mwpm_aware_w = wrong["MWPM[aware]"]
        
        delta_mwpm = mwpm_blind_w.mean() - mwpm_aware_w.mean()
        lo, hi = boot_ci(mwpm_blind_w, mwpm_aware_w, args.bootstrap, rng)
        
        print(f"  Δ_MWPM (blind - aware) = {delta_mwpm:+.5f}")
        if lo is not None:
            print(f"    95% CI: [{lo:+.5f}, {hi:+.5f}]")
    
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    
    print(f"\n✓ saved: {csv_path}")
    print("\n" + "="*80)
    print("Results:")
    print("="*80)
    for r in rows:
        print(f"  {r['decoder']:<30} LER={r['ler']:.4f}")

if __name__ == "__main__":
    main()
