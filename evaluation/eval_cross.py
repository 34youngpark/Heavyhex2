#!/usr/bin/env python3
"""
Cross-evaluation: train profile != test profile
===============================================
train.py는 train/test를 같은 noise profile에서 읽는다(대각선). 이 스크립트는
체크포인트(=학습 profile)와 테스트셋(=평가 profile)을 분리해서
N x N LER 매트릭스를 만든다. "idle noise를 학습한 CNN이 학습 안 한 CNN보다
낫다"는 주장은 이 매트릭스의 **비대각 성분**에서만 나온다.

같은 테스트셋에 대해 4개 디코더를 비교한다:
  * CNN  (idle-aware)   — id>0 profile로 학습한 체크포인트
  * CNN  (idle-blind)   — id=0.0 profile로 학습한 체크포인트
  * MWPM (idle-aware)   — 테스트 profile의 DEM으로 만든 matching
  * MWPM (idle-blind)   — id=0.0 profile의 DEM으로 만든 matching

MWPM 두 개가 반드시 필요하다. CNN 두 개만 비교하면
"train/test 분포가 맞으면 좋다"는 자명한 결과라 리뷰어가 안 받아준다.
CNN의 mismatch penalty가 MWPM의 mismatch penalty보다 작아야 주장이 선다.

사용법:
  # 기본: id=0.0 학습 체크포인트를 blind baseline으로 잡고 전 profile 교차평가
  python eval_cross.py -p 0.005

  # 특정 조합만
  python eval_cross.py -p 0.005 \
      --train-noise realistic/dp0.001_mf0.01_rf0.01_gd0.008_id0.0 \
                    realistic/dp0.001_mf0.01_rf0.01_gd0.008_id0.002 \
      --test-noise  realistic/dp0.001_mf0.01_rf0.01_gd0.008_id0.002

출력: results/cross/cross_p{p}_c{cycles}.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from dataset_generation.heavyhex33_stim import (  # noqa: E402
    noise_tag, DISTANCE, ALL_NOISE, detectors_from_tensor)
from model.data import npz_path  # noqa: E402
from evaluation.metrics import ler, ler_from_logits  # noqa: E402

BLIND = "realistic/dp0.001_mf0.01_rf0.01_gd0.008_id0.0"


def parse_args():
    ap = argparse.ArgumentParser(description="cross-profile LER matrix")
    ap.add_argument("-p", "--rate", type=float, default=0.005)
    ap.add_argument("-e", "--error-type", default="X")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--train-noise", nargs="+", default=None,
                    help="체크포인트의 학습 profile들 (default: ALL_NOISE)")
    ap.add_argument("--test-noise", nargs="+", default=None,
                    help="평가할 테스트셋 profile들 (default: ALL_NOISE)")
    ap.add_argument("--data-dir", default=str(_ROOT / "dataset"))
    ap.add_argument("--ckpt-dir", default=str(_ROOT / "checkpoint"))
    ap.add_argument("--outdir", default=str(_ROOT / "results" / "cross"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-mwpm", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="LER 차이의 부트스트랩 CI 반복수 (0이면 생략)")
    ap.add_argument("--solution", action="store_true")
    return ap.parse_args()


def ckpt_path(ckpt_dir, noise, cycles, p):
    """train.py의 tag 규칙과 동일: CNN_d3_c3_p{p}_{noise_tag}.pt"""
    tag = f"d{DISTANCE}_c{cycles}_p{p}_{noise_tag(noise)}"
    return Path(ckpt_dir) / f"CNN_{tag}.pt"


def load_model(path, cycles, device, use_solution=False):
    if use_solution:
        from solutions.cnn_solution import HeavyHexCNN
    else:
        from model.cnn_skeleton import HeavyHexCNN
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = HeavyHexCNN(in_channels=2 * cycles)
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device).eval()


def cnn_predict(model, features, device, batch=8192):
    """features (N, 2C, 4, 5) uint8 -> logical head 확률/예측"""
    out = []
    with torch.no_grad():
        for i in range(0, features.shape[0], batch):
            xb = torch.from_numpy(
                np.ascontiguousarray(features[i:i + batch])).to(device)
            _, ll = model(xb)
            out.append(ll.float().cpu().numpy().ravel())
    return np.concatenate(out)


def boot_ci(a_wrong, b_wrong, n_boot, rng):
    """LER(a) - LER(b)의 부트스트랩 95% CI. 같은 샷에 짝지어 리샘플."""
    if n_boot <= 0:
        return None, None
    n = a_wrong.shape[0]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a_wrong[idx].mean() - b_wrong[idx].mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_noises = args.train_noise or ALL_NOISE
    test_noises = args.test_noise or ALL_NOISE
    rng = np.random.default_rng(0)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"cross_p{args.rate}_c{args.cycles}.csv"

    # 학습 profile별 모델을 한 번만 로드
    models = {}
    for tn in train_noises:
        cp = ckpt_path(args.ckpt_dir, tn, args.cycles, args.rate)
        if not cp.exists():
            print(f"  [skip] 체크포인트 없음: {cp}")
            continue
        models[tn] = load_model(cp, args.cycles, device, args.solution)
    if not models:
        sys.exit("체크포인트를 하나도 못 찾음. train.py를 먼저 돌릴 것.")

    matching_cache = {}

    def get_matching(profile):
        if profile not in matching_cache:
            from baseline.mwpm import build_matching
            matching_cache[profile] = build_matching(
                args.cycles, args.error_type, args.rate, profile)
        return matching_cache[profile]

    rows = []
    for test_n in test_noises:
        path = npz_path(args.data_dir, test_n, "test",
                        args.cycles, args.rate, args.error_type)
        if not Path(path).exists():
            print(f"  [skip] 테스트셋 없음: {path}")
            continue
        d = np.load(path)
        feats, labels, y = d["features"], d["labels"], d["logical_labels"]
        n_shots = y.shape[0]
        print(f"\n=== test: {noise_tag(test_n)}  (shots={n_shots:,}, "
              f"raw LER={y.mean():.4f}) ===")

        wrong = {}   # 디코더 이름 -> per-shot 오답 마스크 (부트스트랩용)

        # --- CNN 들 ---
        for train_n, model in models.items():
            logits = cnn_predict(model, feats, device)
            pred = (logits > 0).astype(np.uint8)
            name = f"CNN[train={noise_tag(train_n)}]"
            wrong[name] = (pred != y).astype(np.float64)
            rows.append({
                "test_noise": noise_tag(test_n),
                "decoder": name,
                "train_noise": noise_tag(train_n),
                "matched": int(train_n == test_n),
                "ler": float(wrong[name].mean()),
                "raw_ler": float(y.mean()),
                "shots": n_shots,
            })

        # --- MWPM aware / blind ---
        if not args.no_mwpm:
            det = detectors_from_tensor(feats, labels)
            for tag, prof in (("aware", test_n), ("blind", BLIND)):
                pred = np.asarray(
                    get_matching(prof).decode_batch(det), dtype=np.uint8)
                pred = pred.reshape(pred.shape[0], -1)[:, 0]
                name = f"MWPM[{tag}]"
                wrong[name] = (pred != y).astype(np.float64)
                rows.append({
                    "test_noise": noise_tag(test_n),
                    "decoder": name,
                    "train_noise": noise_tag(prof),
                    "matched": int(prof == test_n),
                    "ler": float(wrong[name].mean()),
                    "raw_ler": float(y.mean()),
                    "shots": n_shots,
                })

        # --- 핵심 비교: idle-aware CNN vs idle-blind CNN (같은 테스트셋) ---
        blind_name = f"CNN[train={noise_tag(BLIND)}]"
        aware_name = f"CNN[train={noise_tag(test_n)}]"
        if blind_name in wrong and aware_name in wrong and aware_name != blind_name:
            d_cnn = wrong[aware_name].mean() - wrong[blind_name].mean()
            lo, hi = boot_ci(wrong[aware_name], wrong[blind_name],
                             args.bootstrap, rng)
            print(f"  ΔLER (CNN aware - CNN blind) = {d_cnn:+.5f}"
                  + (f"  95% CI [{lo:+.5f}, {hi:+.5f}]" if lo is not None else ""))
            if not args.no_mwpm:
                d_mwpm = wrong["MWPM[aware]"].mean() - wrong["MWPM[blind]"].mean()
                print(f"  ΔLER (MWPM aware - MWPM blind) = {d_mwpm:+.5f}")
                print(f"  -> CNN이 idle 정보를 MWPM보다 잘 쓰면 d_cnn < d_mwpm")

        for r in rows[-len(wrong):]:
            print(f"    {r['decoder']:<55} LER={r['ler']:.4f}")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved -> {csv_path}")


if __name__ == "__main__":
    main()