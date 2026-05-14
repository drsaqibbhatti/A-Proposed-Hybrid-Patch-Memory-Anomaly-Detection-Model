# AegisAD: Hybrid PatchCore + Student/Autoencoder Anomaly Detector

This is a complete PyTorch anomaly-detection project built for your `AnoDataset` style:

- **Training data:** normal images only.
- **Evaluation data:** defect images, or normal + defect images when you want metrics.
- **Input:** grayscale images loaded as `1 x H x W`; the model internally adapts them for ImageNet-style teachers.
- **Output:** image-level anomaly scores, anomaly maps, overlays, CSV results, metrics JSON, and a saved checkpoint.

## Why this design

AegisAD combines three complementary signals:

1. **PatchCore memory bank** over frozen teacher features.
   - Stores representative normal patch embeddings.
   - At test time, a patch is anomalous if it is far from the normal memory.
   - Strong default for industrial anomaly detection when you have no defect samples.

2. **Student-teacher distillation branch.**
   - A lightweight student learns to predict frozen teacher patch embeddings on normal images.
   - Defects tend to produce teacher features the student has not learned to mimic.

3. **Feature autoencoder branch.**
   - Learns global normal feature structure.
   - Helps with logical/global anomalies that pure local patch matching can miss.

The final score is a calibrated weighted ensemble of these components.

## Files

```text
anomaly/
  datasets.py        # Your AnoDataset-compatible loader + eval loader
  backbones.py       # Frozen teachers: WideResNet/ResNet/EfficientNet/PDN
  pdn.py             # EfficientAD-style PDN-S and PDN-M definitions
  models.py          # AegisAD model
  losses.py          # Hard feature loss, feature MSE, consistency, SSIM utility
  memory.py          # Patch extraction, k-center coreset, nearest-neighbor search
  calibration.py     # Normal-data map normalization and threshold calibration
  metrics.py         # AUROC/AUPRC/F1/confusion metrics
  visualization.py   # Heatmaps and overlays
train.py             # Normal-only training script
eval.py              # Scoring, maps, overlays, metrics
configs/             # Default config notes
scripts/run_example.sh
```

## Installation

Use a matching PyTorch and TorchVision build for your CUDA/Python environment.

```bash
pip install -r requirements.txt
```

If TorchVision is not working in your environment, use the PDN teacher path instead:

```bash
python train.py \
  --train-dir /data/product/train/good \
  --val-normal-dir /data/product/val/good \
  --out-dir runs/product_aegis_pdn \
  --teacher pdn_s \
  --teacher-ckpt /path/to/pdn_imagenet_pretrained.pth \
  --image-size 256
```

## Recommended training command

Use this first if you do not have a PDN checkpoint:

```bash
python train.py \
  --train-dir /data/product/train/good \
  --val-normal-dir /data/product/val/good \
  --out-dir runs/product_aegis \
  --image-size 256 \
  --teacher wide_resnet50_2 \
  --epochs 20 \
  --batch-size 8 \
  --coreset-method greedy \
  --coreset-ratio 0.05 \
  --max-memory-patches 20000
```

If your dataset is small, keep `--coreset-method greedy`. If your dataset is huge and memory selection is slow, switch to:

```bash
--coreset-method random --max-memory-patches 30000
```

## Recommended PDN command

Use this if your PDN checkpoint is already ImageNet-pretrained:

```bash
python train.py \
  --train-dir /data/product/train/good \
  --val-normal-dir /data/product/val/good \
  --out-dir runs/product_aegis_pdn \
  --image-size 256 \
  --teacher pdn_s \
  --teacher-ckpt /path/to/pdn_pretrained.pth \
  --epochs 30 \
  --batch-size 16 \
  --target-dim 256
```

For many industrial grayscale tasks, try both `wide_resnet50_2` and `pdn_s`. Keep the one with better normal/defect separation on a validation set.

## Evaluation on defect images only

This produces scores, predictions using the calibrated threshold, maps, overlays, and a CSV. AUROC/AUPRC are not defined unless you also provide normal test images.

```bash
python eval.py \
  --ckpt runs/product_aegis/best_model.pt \
  --defect-dir /data/product/test/defect \
  --out-dir runs/product_aegis/eval_defects \
  --save-maps \
  --save-overlays
```

## Evaluation with normal + defect images

This computes AUROC, AUPRC, best F1, and threshold confusion metrics.

```bash
python eval.py \
  --ckpt runs/product_aegis/best_model.pt \
  --normal-dir /data/product/test/good \
  --defect-dir /data/product/test/defect \
  --out-dir runs/product_aegis/eval_labeled \
  --save-maps \
  --save-overlays
```

## Outputs

Training writes:

```text
runs/product_aegis/
  best_model.pt
  calibration.json
  args.json
```

Evaluation writes:

```text
runs/product_aegis/eval_labeled/
  results.csv
  metrics.json
  maps/*.png
  overlays/*.png
```

`results.csv` contains filename, anomaly score, prediction, threshold, component scores, and map paths.

## Important tuning notes

### Threshold

The default threshold is the `99.5%` quantile of normal calibration scores. Best practice is to pass a clean normal validation folder via `--val-normal-dir`. Without that, the script calibrates on the training images, which can be slightly optimistic.

To make the model stricter:

```bash
--threshold-quantile 0.999
```

To override at evaluation:

```bash
python eval.py --ckpt runs/product_aegis/best_model.pt --defect-dir ... --threshold 2.75
```

### Image size

Start with `256`. For tiny defects, try `384` or `512`, but expect more memory and slower nearest-neighbor search.

### Memory bank

More memory patches can improve recall but slow inference.

Good starting points:

```bash
--max-memory-patches 10000   # fast
--max-memory-patches 20000   # balanced default
--max-memory-patches 50000   # stronger but slower
```

### Component weights

Default:

```bash
--w-patchcore 0.60 --w-student 0.25 --w-autoencoder 0.15
```

If defects are tiny/local, increase PatchCore:

```bash
--w-patchcore 0.80 --w-student 0.15 --w-autoencoder 0.05
```

If defects are logical/global, increase the autoencoder branch:

```bash
--w-patchcore 0.45 --w-student 0.25 --w-autoencoder 0.30
```

### Pure PatchCore mode

For a simpler and often very strong baseline:

```bash
python train.py \
  --train-dir /data/product/train/good \
  --val-normal-dir /data/product/val/good \
  --out-dir runs/product_patchcore \
  --disable-student \
  --disable-autoencoder \
  --teacher wide_resnet50_2 \
  --image-size 256
```

## Using your own dataloader

The included `AnoDataset` is intentionally compatible with the dataloader you posted. If you want to use your exact class, replace the import in `train.py` and keep the transform behavior the same: return a `1 x H x W` float tensor in `[0, 1]`.

## Practical recipe for best results

1. Train `wide_resnet50_2` hybrid at image size 256.
2. Train `pdn_s` hybrid using your ImageNet-pretrained PDN.
3. Evaluate both on `normal-dir + defect-dir` if possible.
4. Use overlays to inspect false positives and false negatives.
5. Increase image size only if tiny defects are missed.
6. Tune threshold on clean normal validation images, not on training images.


