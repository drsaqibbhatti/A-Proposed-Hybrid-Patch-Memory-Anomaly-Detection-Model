import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly.calibration import calibrate_model
from anomaly.datasets import AnoDataset, build_transform
from anomaly.losses import weighted_training_loss
from anomaly.memory import build_memory_bank, collect_patch_embeddings
from anomaly.models import AegisAD
from anomaly.utils import count_parameters, ensure_dir, get_device, save_json, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Train AegisAD on normal images only.")
    p.add_argument("--train-dir", required=True, help="Directory containing normal training images.")
    p.add_argument("--val-normal-dir", default=None, help="Optional normal validation/calibration directory.")
    p.add_argument("--out-dir", default="runs/aegis_ad", help="Output directory.")

    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--teacher", default="wide_resnet50_2", choices=["wide_resnet50_2", "resnet50", "resnet34", "resnet18", "efficientnet_b0", "pdn_s", "pdn_m"])
    p.add_argument("--teacher-ckpt", default=None, help="Optional checkpoint for PDN/custom teacher weights.")
    p.add_argument("--no-pretrained", action="store_true", help="Do not load torchvision ImageNet weights.")
    p.add_argument("--layers", nargs="*", default=None, help="Teacher feature layers. Default depends on teacher.")
    p.add_argument("--target-dim", type=int, default=256, help="Projected teacher embedding dimension.")
    p.add_argument("--local-agg-kernel", type=int, default=3, help="Patch feature local aggregation kernel.")
    p.add_argument("--projector-seed", type=int, default=42)
    p.add_argument("--pdn-out-channels", type=int, default=384)
    p.add_argument("--pdn-padding", default=True, action=argparse.BooleanOptionalAction, help="Enable BatchNorm in PDN teacher (default: True). Use --no-pdn-padding to disable.")

    p.add_argument("--disable-student", action="store_true")
    p.add_argument("--disable-autoencoder", action="store_true")
    p.add_argument("--student-width", type=int, default=96)
    p.add_argument("--ae-width", type=int, default=96)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hard-ratio", type=float, default=0.10)
    p.add_argument("--student-weight", type=float, default=1.0)
    p.add_argument("--autoencoder-weight", type=float, default=0.5)
    p.add_argument("--consistency-weight", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")

    p.add_argument("--coreset-ratio", type=float, default=0.05)
    p.add_argument("--coreset-method", default="greedy", choices=["greedy", "random"])
    p.add_argument("--max-train-patches", type=int, default=200000)
    p.add_argument("--max-memory-patches", type=int, default=20000)
    p.add_argument("--nn-chunk-size", type=int, default=2048)
    p.add_argument("--memory-chunk-size", type=int, default=20000)

    p.add_argument("--topk-ratio", type=float, default=0.01)
    p.add_argument("--threshold-quantile", type=float, default=0.995)
    p.add_argument("--w-patchcore", type=float, default=0.60)
    p.add_argument("--w-student", type=float, default=0.25)
    p.add_argument("--w-autoencoder", type=float, default=0.15)

    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_loader(path: str, image_size: int, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    ds = AnoDataset(path, transform=build_transform(image_size), return_filename=False)
    if len(ds) == 0:
        raise RuntimeError(f"No images found in {path}")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def train_distillation_branches(model: AegisAD, loader: DataLoader, args, device: torch.device) -> None:
    params = []
    if model.student is not None:
        params += list(model.student.parameters())
    if model.autoencoder is not None:
        params += list(model.autoencoder.parameters())
    if not params or args.epochs <= 0:
        print("Skipping student/autoencoder training.")
        return

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"Training distillation branches for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running: Dict[str, float] = {}
        count = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images in pbar:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                target, student_pred, ae_pred = model.training_predictions(images)
                losses = weighted_training_loss(
                    student_pred=student_pred,
                    ae_pred=ae_pred,
                    target=target,
                    hard_ratio=args.hard_ratio,
                    student_weight=args.student_weight,
                    autoencoder_weight=args.autoencoder_weight,
                    consistency_weight=args.consistency_weight,
                )
            scaler.scale(losses["total"]).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bs = images.shape[0]
            count += bs
            for key, val in losses.items():
                running[key] = running.get(key, 0.0) + float(val.detach().cpu().item()) * bs
            pbar.set_postfix({k: f"{v / max(count, 1):.5f}" for k, v in running.items()})
        scheduler.step()


def save_checkpoint(model: AegisAD, args, out_dir: Path, calibration: Dict) -> Path:
    ckpt_path = out_dir / "best_model.pt"
    memory = None if model.memory_bank is None else model.memory_bank.detach().cpu()
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "memory_bank": memory,
        "calibration": calibration,
        "config": {
            "model": model.model_config,
            "training_args": vars(args),
        },
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def main():
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.out_dir)
    save_json(vars(args), out_dir / "args.json")
    device = get_device(args.device)
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
        print(f"  VRAM total : {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        print(f"  VRAM free  : {torch.cuda.mem_get_info(device)[0] / 1024**3:.1f} GB")
    else:
        print(f"Using device: {device}")

    train_loader = build_loader(args.train_dir, args.image_size, args.batch_size, args.num_workers, shuffle=True)
    memory_loader = build_loader(args.train_dir, args.image_size, args.batch_size, args.num_workers, shuffle=False)
    calib_dir = args.val_normal_dir if args.val_normal_dir else args.train_dir
    calib_loader = build_loader(calib_dir, args.image_size, args.batch_size, args.num_workers, shuffle=False)

    if args.teacher in ("pdn_s", "pdn_m") and not args.teacher_ckpt:
        print("Warning: PDN teacher selected without --teacher-ckpt. A random PDN teacher is not recommended.")

    model = AegisAD(
        teacher_name=args.teacher,
        pretrained=not args.no_pretrained,
        teacher_ckpt=args.teacher_ckpt,
        layers=args.layers,
        target_dim=args.target_dim,
        local_agg_kernel=args.local_agg_kernel,
        enable_student=not args.disable_student,
        enable_autoencoder=not args.disable_autoencoder,
        student_width=args.student_width,
        ae_width=args.ae_width,
        image_size=args.image_size,
        projector_seed=args.projector_seed,
        pdn_out_channels=args.pdn_out_channels,
        pdn_padding=args.pdn_padding,
    ).to(device)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")

    train_distillation_branches(model, train_loader, args, device)

    model.eval()
    patches = collect_patch_embeddings(
        model,
        memory_loader,
        device=device,
        max_train_patches=args.max_train_patches,
        seed=args.seed,
    )
    print(f"Candidate normal patches: {patches.shape[0]:,} x {patches.shape[1]}")
    memory = build_memory_bank(
        patches,
        coreset_ratio=args.coreset_ratio,
        max_memory_patches=args.max_memory_patches,
        method=args.coreset_method,
        device=device,
        seed=args.seed,
    )
    print(f"Memory bank size: {memory.shape[0]:,} x {memory.shape[1]}")
    model.set_memory(memory)
    model.move_memory(device)

    weights = {
        "patchcore": args.w_patchcore,
        "student": args.w_student,
        "autoencoder": args.w_autoencoder,
    }
    calibration = calibrate_model(
        model,
        calib_loader,
        device=device,
        weights=weights,
        topk_ratio=args.topk_ratio,
        threshold_quantile=args.threshold_quantile,
        seed=args.seed,
        nn_chunk_size=args.nn_chunk_size,
        memory_chunk_size=args.memory_chunk_size,
    )
    save_json(calibration, out_dir / "calibration.json")
    ckpt_path = save_checkpoint(model, args, out_dir, calibration)
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Suggested threshold: {calibration['combined']['threshold']:.6f}")


if __name__ == "__main__":
    main()
