import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly.datasets import LabeledAnoDataset, build_transform, collate_eval
from anomaly.metrics import compute_image_metrics
from anomaly.models import AegisAD
from anomaly.utils import ensure_dir, get_device, save_json, seed_everything
from anomaly.visualization import save_anomaly_images


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AegisAD and export scores/maps.")
    p.add_argument("--ckpt", required=True, help="Path to best_model.pt from train.py")
    p.add_argument("--data-dir", default=None, help="Unlabeled directory. Labels will be -1.")
    p.add_argument("--normal-dir", default=None, help="Optional normal test directory, label=0.")
    p.add_argument("--defect-dir", default=None, help="Optional defect test directory, label=1.")
    p.add_argument("--out-dir", default="runs/aegis_ad/eval")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--threshold", type=float, default=None, help="Override checkpoint threshold.")
    p.add_argument("--save-maps", action="store_true")
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--no-normalize", action="store_true", help="Use raw uncalibrated maps/scores.")
    p.add_argument("--nn-chunk-size", type=int, default=2048)
    p.add_argument("--memory-chunk-size", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if not (args.data_dir or args.normal_dir or args.defect_dir):
        raise SystemExit("Provide at least one of --data-dir, --normal-dir, or --defect-dir")

    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = get_device(args.device)
    model, payload = AegisAD.from_checkpoint(args.ckpt, map_location="cpu")
    model.to(device).eval()
    model.move_memory(device)

    image_size = payload.get("config", {}).get("model", {}).get("image_size", 256)
    ds = LabeledAnoDataset(
        normal_dir=args.normal_dir,
        defect_dir=args.defect_dir,
        data_dir=args.data_dir,
        transform=build_transform(image_size),
    )
    if len(ds) == 0:
        raise RuntimeError("No images found for evaluation")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_eval,
        persistent_workers=(args.num_workers > 0),
    )

    threshold = args.threshold
    if threshold is None:
        threshold = model.calibration.get("combined", {}).get("threshold")

    rows = []
    labels_for_metrics = []
    scores_for_metrics = []

    with torch.no_grad():
        sample_idx = 0
        for images, labels, names, paths in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            out = model.predict(
                images,
                normalize=not args.no_normalize,
                nn_chunk_size=args.nn_chunk_size,
                memory_chunk_size=args.memory_chunk_size,
            )
            scores = out["score"].detach().cpu().tolist()
            amap = out["anomaly_map"].detach().cpu()
            comp_score_names = sorted([k for k in out.keys() if k.endswith("_score") and k != "score"])

            for i, (name, path, label, score) in enumerate(zip(names, paths, labels.tolist(), scores)):
                pred = int(score >= threshold) if threshold is not None else -1
                stem = f"{sample_idx:06d}_{Path(name).stem}"
                map_path = None
                overlay_path = None
                if args.save_maps or args.save_overlays:
                    map_path, overlay_path = save_anomaly_images(
                        images[i].detach().cpu(),
                        amap[i],
                        stem=stem,
                        out_dir=out_dir,
                        save_map=args.save_maps,
                        save_overlay=args.save_overlays,
                    )
                row = {
                    "index": sample_idx,
                    "filename": name,
                    "path": path,
                    "label": int(label),
                    "score": float(score),
                    "threshold": float(threshold) if threshold is not None else "",
                    "pred": pred,
                    "map_path": map_path or "",
                    "overlay_path": overlay_path or "",
                }
                for key in comp_score_names:
                    row[key] = float(out[key][i].detach().cpu().item())
                rows.append(row)
                labels_for_metrics.append(int(label))
                scores_for_metrics.append(float(score))
                sample_idx += 1

    csv_path = out_dir / "results.csv"
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    preferred = ["index", "filename", "path", "label", "score", "threshold", "pred", "patchcore_score", "student_score", "autoencoder_score", "map_path", "overlay_path"]
    fieldnames = [f for f in preferred if f in fieldnames] + [f for f in fieldnames if f not in preferred]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = compute_image_metrics(labels_for_metrics, scores_for_metrics, threshold=threshold)
    save_json(metrics, out_dir / "metrics.json")
    print(f"Saved results: {csv_path}")
    if threshold is not None:
        print(f"Threshold: {threshold:.6f}")
    print(f"Metrics: {metrics}")
    if metrics.get("num_normal", 0) == 0 or metrics.get("num_defect", 0) == 0:
        print("AUROC/AUPRC need both normal and defect images. With defect-only eval, use scores, maps, and threshold predictions.")


if __name__ == "__main__":
    main()
