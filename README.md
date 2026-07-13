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

## Smoke commands

```bash
python3 -m geo_vqvae.train_top --prepared-dir geo_vqvae/prepared --output-dir geo_vqvae/runs/top_smoke --epochs 1 --batch-size 2 --sequence-length 128 --max-sequences 2 --d-model 64 --n-heads 4 --n-layers 2 --device cpu --no-progress

python3 -m geo_vqvae.train_top_prior --prepared-dir geo_vqvae/prepared --top-checkpoint geo_vqvae/runs/top_smoke/best_top.pt --output-dir geo_vqvae/runs/top_prior_smoke --epochs 1 --batch-size 2 --sequence-length 128 --max-sequences 2 --d-model 64 --n-heads 4 --n-layers 2 --device cpu --no-progress

python3 -m geo_vqvae.train_low --prepared-dir geo_vqvae/prepared --top-checkpoint geo_vqvae/runs/top_smoke/best_top.pt --top-prior-checkpoint geo_vqvae/runs/top_prior_smoke/best_top_prior.pt --output-dir geo_vqvae/runs/low_smoke --epochs 1 --batch-size 2 --sequence-length 128 --max-sequences 2 --d-model 64 --n-heads 4 --n-layers 2 --device cpu --no-progress
```
