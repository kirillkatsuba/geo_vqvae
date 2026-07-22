# geo_vqvae

Block-as-token VQ-VAE2 prototype for assay-conditioned block model generation.

The first version uses the same assay feature family as the previous
microblock models:

- chemistry: `Au_Final`, non-target ICP chemistry, sulfur/carbon variants;
- lithology/categorical fields: `LITH`, `LITH_STRUCTURE`, `REDOX`,
  `ALTERATION`, `MINERAL_GP`, `TYPE_MINERAL_GP`, etc.;
- coordinates `X/Y/Z`.

The model is hierarchical:

```text
assay tokens -> TopVQTransformer -> discrete top codes
block features + previous top codes -> TopPriorTransformer -> generated top codes
block features + generated top context -> LowVQVAE2 -> AS/S/CORG-1/CA/FE
```

`TopPriorTransformer` is the piece that starts replacing raw KNN chemistry at
inference time. KNN/nearest-assay assignment is used only to create supervised
top-code labels for training the top prior.

## 1. Prepare data

```bash
python3 -m geo_vqvae.prepare_data \
  --root . \
  --output-dir geo_vqvae/prepared \
  --val-fraction 0.15 \
  --val-axis Y \
  --val-side high
```

Outputs:

- `assays.parquet`
- `center_blocks.parquet`
- `north_blocks.parquet`
- `assay_preprocessor.json`
- `block_preprocessor.json`
- `target_scaler.json`
- `metadata.json`

## 2. Train top-level VQ model

```bash
python3 -m geo_vqvae.train_top \
  --prepared-dir geo_vqvae/prepared \
  --output-dir geo_vqvae/runs/top_v1 \
  --epochs 100 \
  --batch-size 4 \
  --sequence-length 1024 \
  --d-model 256 \
  --n-heads 8 \
  --n-layers 4 \
  --codebook-size 256 \
  --learning-rate 1e-4 \
  --device cuda
```

Output:

```text
geo_vqvae/runs/top_v1/best_top.pt
```

## 3. Train autoregressive top prior

```bash
python3 -m geo_vqvae.train_top_prior \
  --prepared-dir geo_vqvae/prepared \
  --top-checkpoint geo_vqvae/runs/top_v1/best_top.pt \
  --output-dir geo_vqvae/runs/top_prior_v1 \
  --epochs 100 \
  --batch-size 4 \
  --sequence-length 1024 \
  --d-model 256 \
  --n-heads 8 \
  --n-layers 4 \
  --learning-rate 1e-4 \
  --device cuda
```

Output:

```text
geo_vqvae/runs/top_prior_v1/best_top_prior.pt
```

## 4. Train low-level VQ-VAE2

Recommended mode: condition low-level generation on generated top-prior codes.

```bash
python3 -m geo_vqvae.train_low \
  --prepared-dir geo_vqvae/prepared \
  --top-checkpoint geo_vqvae/runs/top_v1/best_top.pt \
  --top-prior-checkpoint geo_vqvae/runs/top_prior_v1/best_top_prior.pt \
  --output-dir geo_vqvae/runs/low_v1_prior \
  --epochs 150 \
  --batch-size 4 \
  --sequence-length 1024 \
  --d-model 256 \
  --n-heads 8 \
  --n-layers 6 \
  --codebook-size 256 \
  --lambda-code 0.2 \
  --lambda-corr 0.1 \
  --learning-rate 1e-4 \
  --device cuda
```

The loss includes:

```text
target reconstruction loss
+ VQ commitment/codebook loss
+ low-code prior CE loss
+ correlation matrix regularization
```

Correlation regularization modes:

- `--corr-mode batch`: previous behavior, match the correlation matrix of each
  predicted batch to the ground-truth targets from the same batch.
- `--corr-mode global`: match the correlation matrix of predictions to global
  reference correlations from the source block model and assays.
- `--corr-mode batch_global`: use both constraints.

For the global geological constraint:

```bash
python3 -m geo_vqvae.train_low \
  --prepared-dir geo_vqvae/prepared \
  --top-checkpoint geo_vqvae/runs/top_v1/best_top.pt \
  --top-prior-checkpoint geo_vqvae/runs/top_prior_v1/best_top_prior.pt \
  --output-dir geo_vqvae/runs/low_v1_prior_global_corr \
  --epochs 60 \
  --batch-size 8 \
  --sequence-length 512 \
  --d-model 256 \
  --n-heads 8 \
  --n-layers 6 \
  --codebook-size 128 \
  --lambda-vq 0.05 \
  --lambda-code 0.03 \
  --lambda-corr 0.10 \
  --corr-mode global \
  --corr-reference-split train \
  --corr-block-weight 1.0 \
  --corr-assay-weight 1.0 \
  --learning-rate 3e-5 \
  --device cuda
```

Output:

```text
geo_vqvae/runs/low_v1_prior/best_low.pt
```

## 5. Evaluate

```bash
python3 -m geo_vqvae.evaluate \
  --prepared-dir geo_vqvae/prepared \
  --low-checkpoint geo_vqvae/runs/low_v1_prior/best_low.pt \
  --output-dir geo_vqvae/eval/low_v1_prior \
  --domain both \
  --sequence-length 1024 \
  --batch-size 4 \
  --device cuda
```

Outputs per domain:

- `predictions.csv`
- `metrics.csv`

### Inference-time correlation matching

This is an alternative to correlation regularization: the model checkpoint is
not retrained. After prediction, the vector `AS/S/CORG-1/CA/FE` is linearly
recolored so that its cross-target correlation matrix is closer to a reference
matrix from assays, center block model, or their blend.

Recommended first check for the current `v7_soft_t4` setup:

```bash
python3 -m geo_vqvae.evaluate \
  --prepared-dir geo_vqvae/prepared_v2 \
  --low-checkpoint geo_vqvae/runs/low_v7_soft_val/best_low.pt \
  --output-dir geo_vqvae/eval/low_v7_soft_val_t4_corr_assay_s05 \
  --domain both \
  --sequence-length 512 \
  --batch-size 64 \
  --decode-mode soft \
  --softmax-temperature 4.0 \
  --corr-adjust assay \
  --corr-adjust-strength 0.5 \
  --keep-unadjusted-predictions \
  --device cuda
```

Useful variants:

```bash
# Match training assay correlations more strongly.
--corr-adjust assay --corr-adjust-strength 1.0

# Match center block-model correlations.
--corr-adjust block --corr-adjust-block-split train --corr-adjust-strength 0.5

# Blend assay and center block-model references.
--corr-adjust blend --corr-adjust-assay-weight 2.0 --corr-adjust-block-weight 1.0 --corr-adjust-strength 0.5
```

Generate the full north block model with the same post-calibration:

```bash
python3 -m geo_vqvae.predict_north_blocks \
  --prepared-dir geo_vqvae/prepared_v2 \
  --low-checkpoint geo_vqvae/runs/low_v7_soft_val/best_low.pt \
  --output-csv geo_vqvae/predictions/north_blocks_low_v7_soft_t4_corr_assay_s05.csv \
  --sequence-length 512 \
  --batch-size 64 \
  --decode-mode soft \
  --softmax-temperature 4.0 \
  --corr-adjust assay \
  --corr-adjust-strength 0.5 \
  --device cuda
```

## Smoke commands

```bash
python3 -m geo_vqvae.train_top --prepared-dir geo_vqvae/prepared --output-dir geo_vqvae/runs/top_smoke --epochs 1 --batch-size 2 --sequence-length 128 --max-sequences 2 --d-model 64 --n-heads 4 --n-layers 2 --device cpu --no-progress

python3 -m geo_vqvae.train_top_prior --prepared-dir geo_vqvae/prepared --top-checkpoint geo_vqvae/runs/top_smoke/best_top.pt --output-dir geo_vqvae/runs/top_prior_smoke --epochs 1 --batch-size 2 --sequence-length 128 --max-sequences 2 --d-model 64 --n-heads 4 --n-layers 2 --device cpu --no-progress

python3 -m geo_vqvae.train_low --prepared-dir geo_vqvae/prepared --top-checkpoint geo_vqvae/runs/top_smoke/best_top.pt --top-prior-checkpoint geo_vqvae/runs/top_prior_smoke/best_top_prior.pt --output-dir geo_vqvae/runs/low_smoke --epochs 1 --batch-size 2 --sequence-length 128 --max-sequences 2 --d-model 64 --n-heads 4 --n-layers 2 --device cpu --no-progress
```
