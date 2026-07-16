from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .columns import TARGET_COLUMNS
from .evaluate import choose_device, load_low_model, predict_domain
from .preprocessing import TargetScaler
from .top_context import (
    attach_prior_top_context,
    attach_top_context,
    encode_assay_embeddings,
    load_top_model,
    load_top_prior,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an interactive HTML map with source center/south BM and generated north BM."
    )
    parser.add_argument("--prepared-dir", type=Path, default=Path("geo_vqvae/prepared_v2"))
    parser.add_argument("--low-checkpoint", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, default=Path("geo_vqvae/eval/block_model_continuation.html"))
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-points-per-layer", type=int, default=200000)
    parser.add_argument("--targets", type=str, default=",".join(TARGET_COLUMNS))
    parser.add_argument("--decode-mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--no-north-known-true",
        action="store_true",
        help="Do not include known north BM values as a reference layer.",
    )
    return parser.parse_args()


def sample_frame(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(len(df), size=max_points, replace=False))
    return df.iloc[index].reset_index(drop=True)


def finite_layer_frame(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    columns = ["X", "Y", *value_columns]
    out = df[columns].copy()
    out = out[np.isfinite(out["X"].to_numpy(dtype=float)) & np.isfinite(out["Y"].to_numpy(dtype=float))]
    keep = np.zeros(len(out), dtype=bool)
    for col in value_columns:
        keep |= np.isfinite(out[col].to_numpy(dtype=float))
    return out.loc[keep].reset_index(drop=True)


def round_values(values: np.ndarray, decimals: int = 6) -> list[float | None]:
    values = values.astype(float)
    rounded = np.round(values, decimals=decimals)
    return [None if not np.isfinite(value) else float(value) for value in rounded]


def make_layer(
    layer_id: str,
    label: str,
    df: pd.DataFrame,
    targets: list[str],
    value_prefix: str,
    max_points: int,
    seed: int,
    marker: str,
    default_visible: bool = True,
) -> dict:
    value_columns = [f"{value_prefix}{target}" for target in targets]
    frame = finite_layer_frame(df, value_columns)
    frame = sample_frame(frame, max_points, seed)
    values = {}
    for target, col in zip(targets, value_columns):
        values[target] = round_values(frame[col].to_numpy(dtype=float))
    return {
        "id": layer_id,
        "label": label,
        "marker": marker,
        "visible": default_visible,
        "count": int(len(frame)),
        "x": round_values(frame["X"].to_numpy(dtype=float), decimals=3),
        "y": round_values(frame["Y"].to_numpy(dtype=float), decimals=3),
        "values": values,
    }


def compute_domains(layers: list[dict], targets: list[str]) -> dict[str, list[float]]:
    domains = {}
    for target in targets:
        values = []
        for layer in layers:
            values.extend(value for value in layer["values"][target] if value is not None)
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            domains[target] = [0.0, 1.0]
            continue
        low = float(np.nanpercentile(arr, 2))
        high = float(np.nanpercentile(arr, 98))
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            low = float(np.nanmin(arr))
            high = float(np.nanmax(arr))
        if low == high:
            high = low + 1.0
        domains[target] = [low, high]
    return domains


def attach_context(
    blocks: pd.DataFrame,
    ckpt: dict,
    prepared_dir: Path,
    device: torch.device,
) -> pd.DataFrame:
    assays = pd.read_parquet(prepared_dir / "assays.parquet")
    top_model, top_features = load_top_model(Path(ckpt["top_checkpoint"]), device)
    top_prior_path = ckpt.get("top_prior_checkpoint", "")
    if top_prior_path:
        top_prior, top_prior_ckpt = load_top_prior(Path(top_prior_path), device)
        out, _ = attach_prior_top_context(
            blocks.reset_index(drop=True),
            prior=top_prior,
            top_model=top_model,
            block_feature_columns=ckpt["block_feature_columns"],
            sequence_length=int(top_prior_ckpt["model_config"]["sequence_length"]),
            device=device,
        )
        return out

    _, assay_embeddings = encode_assay_embeddings(assays, top_features, top_model, device)
    out, _ = attach_top_context(
        blocks.reset_index(drop=True),
        assays,
        assay_embeddings,
        k=ckpt["model_config"]["top_k"],
    )
    return out


def render_html(payload: dict) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Block Model Continuation</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f7f8fa; color: #1f2933; }}
    header {{ padding: 14px 18px; border-bottom: 1px solid #d8dee8; background: white; }}
    h1 {{ margin: 0 0 6px; font-size: 20px; }}
    .meta {{ color: #5b6675; font-size: 13px; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center; padding: 12px 18px; background: #eef2f6; border-bottom: 1px solid #d8dee8; }}
    select, button {{ font-size: 14px; padding: 5px 8px; }}
    label {{ font-size: 14px; }}
    main {{ display: grid; grid-template-columns: 1fr 280px; gap: 0; min-height: calc(100vh - 120px); }}
    #canvasWrap {{ position: relative; background: white; }}
    canvas {{ width: 100%; height: 100%; display: block; }}
    aside {{ border-left: 1px solid #d8dee8; background: white; padding: 14px; overflow: auto; }}
    .layer {{ margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #edf0f4; }}
    .layer-title {{ font-weight: 700; margin-bottom: 4px; }}
    .stat {{ font-size: 12px; color: #4c5664; line-height: 1.45; }}
    .swatch {{ display: inline-block; width: 11px; height: 11px; margin-right: 6px; vertical-align: -1px; border: 1px solid #3335; }}
    .hint {{ font-size: 12px; color: #697586; line-height: 1.45; margin-top: 10px; }}
  </style>
</head>
<body>
  <header>
    <h1>Block Model: source BM and generated north continuation</h1>
    <div class="meta" id="meta"></div>
  </header>
  <div class="controls">
    <label>Target <select id="target"></select></label>
    <span id="layerControls"></span>
    <button id="reset">Reset view</button>
  </div>
  <main>
    <div id="canvasWrap"><canvas id="plot"></canvas></div>
    <aside>
      <div id="legend"></div>
      <div class="hint">
        The north prediction layer is generated from the low checkpoint. If the checkpoint has
        a top-prior path, top context is generated autoregressively before low-level target prediction.
        Scroll to zoom, drag to pan.
      </div>
    </aside>
  </main>
  <script>
    const DATA = {payload_json};
    const canvas = document.getElementById('plot');
    const ctx = canvas.getContext('2d');
    const targetSelect = document.getElementById('target');
    const layerControls = document.getElementById('layerControls');
    const legend = document.getElementById('legend');
    const meta = document.getElementById('meta');
    let view = null;
    let dragging = false;
    let lastMouse = null;

    for (const target of DATA.targets) {{
      const option = document.createElement('option');
      option.value = target;
      option.textContent = target;
      targetSelect.appendChild(option);
    }}

    for (const layer of DATA.layers) {{
      const label = document.createElement('label');
      label.style.marginRight = '12px';
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.checked = layer.visible;
      checkbox.dataset.layer = layer.id;
      checkbox.addEventListener('change', draw);
      label.appendChild(checkbox);
      label.append(' ' + layer.label);
      layerControls.appendChild(label);
    }}

    meta.textContent = `checkpoint: ${{DATA.checkpoint}} | points shown: ${{DATA.layers.map(l => l.label + '=' + l.count).join(', ')}}`;

    function colorStops(t) {{
      const stops = [
        [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]
      ];
      t = Math.max(0, Math.min(1, t));
      const p = t * (stops.length - 1);
      const i = Math.min(stops.length - 2, Math.floor(p));
      const f = p - i;
      const a = stops[i], b = stops[i + 1];
      const r = Math.round(a[0] + (b[0] - a[0]) * f);
      const g = Math.round(a[1] + (b[1] - a[1]) * f);
      const bl = Math.round(a[2] + (b[2] - a[2]) * f);
      return `rgb(${{r}},${{g}},${{bl}})`;
    }}

    function currentVisibleIds() {{
      return Array.from(layerControls.querySelectorAll('input[type=checkbox]'))
        .filter(cb => cb.checked)
        .map(cb => cb.dataset.layer);
    }}

    function initView() {{
      const xs = [], ys = [];
      for (const layer of DATA.layers) {{
        xs.push(...layer.x.filter(v => v !== null));
        ys.push(...layer.y.filter(v => v !== null));
      }}
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const padX = (maxX - minX) * 0.04 || 1;
      const padY = (maxY - minY) * 0.04 || 1;
      view = {{ minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY }};
    }}

    function resize() {{
      const rect = canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(400, Math.floor(rect.width * dpr));
      canvas.height = Math.max(400, Math.floor(rect.height * dpr));
      draw();
    }}

    function project(x, y) {{
      const px = (x - view.minX) / (view.maxX - view.minX) * canvas.width;
      const py = canvas.height - (y - view.minY) / (view.maxY - view.minY) * canvas.height;
      return [px, py];
    }}

    function unproject(px, py) {{
      const x = view.minX + px / canvas.width * (view.maxX - view.minX);
      const y = view.minY + (canvas.height - py) / canvas.height * (view.maxY - view.minY);
      return [x, y];
    }}

    function layerStats(layer, target) {{
      const vals = layer.values[target].filter(v => v !== null && Number.isFinite(v));
      if (!vals.length) return 'no finite values';
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const min = Math.min(...vals);
      const max = Math.max(...vals);
      return `n=${{vals.length}}, mean=${{mean.toFixed(4)}}, min=${{min.toFixed(4)}}, max=${{max.toFixed(4)}}`;
    }}

    function drawLegend(target) {{
      const [lo, hi] = DATA.domains[target];
      legend.innerHTML = '';
      const scale = document.createElement('div');
      scale.className = 'layer';
      scale.innerHTML = `<div class="layer-title">${{target}} color scale</div>
        <div class="stat">p2=${{lo.toFixed(4)}} | p98=${{hi.toFixed(4)}}</div>`;
      legend.appendChild(scale);
      for (const layer of DATA.layers) {{
        const item = document.createElement('div');
        item.className = 'layer';
        item.innerHTML = `<div class="layer-title"><span class="swatch" style="background:${{layer.marker === 'square' ? '#111827' : '#ffffff'}}"></span>${{layer.label}}</div>
          <div class="stat">${{layerStats(layer, target)}}</div>`;
        legend.appendChild(item);
      }}
    }}

    function draw() {{
      if (!view) initView();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const target = targetSelect.value || DATA.targets[0];
      const [lo, hi] = DATA.domains[target];
      const visible = new Set(currentVisibleIds());
      for (const layer of DATA.layers) {{
        if (!visible.has(layer.id)) continue;
        const values = layer.values[target];
        const size = layer.marker === 'square' ? 2.1 : 1.9;
        for (let i = 0; i < layer.x.length; i++) {{
          const x = layer.x[i], y = layer.y[i], value = values[i];
          if (x === null || y === null || value === null) continue;
          const [px, py] = project(x, y);
          if (px < -5 || py < -5 || px > canvas.width + 5 || py > canvas.height + 5) continue;
          const t = (value - lo) / (hi - lo);
          ctx.fillStyle = colorStops(t);
          if (layer.marker === 'square') {{
            ctx.fillRect(px - size, py - size, size * 2, size * 2);
          }} else {{
            ctx.beginPath();
            ctx.arc(px, py, size, 0, Math.PI * 2);
            ctx.fill();
          }}
        }}
      }}
      drawLegend(target);
    }}

    targetSelect.addEventListener('change', draw);
    document.getElementById('reset').addEventListener('click', () => {{ initView(); draw(); }});
    canvas.addEventListener('mousedown', e => {{ dragging = true; lastMouse = [e.offsetX, e.offsetY]; }});
    window.addEventListener('mouseup', () => {{ dragging = false; lastMouse = null; }});
    canvas.addEventListener('mousemove', e => {{
      if (!dragging || !lastMouse) return;
      const dpr = window.devicePixelRatio || 1;
      const [x0, y0] = unproject(lastMouse[0] * dpr, lastMouse[1] * dpr);
      const [x1, y1] = unproject(e.offsetX * dpr, e.offsetY * dpr);
      const dx = x0 - x1, dy = y0 - y1;
      view.minX += dx; view.maxX += dx; view.minY += dy; view.maxY += dy;
      lastMouse = [e.offsetX, e.offsetY];
      draw();
    }});
    canvas.addEventListener('wheel', e => {{
      e.preventDefault();
      const dpr = window.devicePixelRatio || 1;
      const [cx, cy] = unproject(e.offsetX * dpr, e.offsetY * dpr);
      const factor = e.deltaY < 0 ? 0.85 : 1.18;
      view.minX = cx + (view.minX - cx) * factor;
      view.maxX = cx + (view.maxX - cx) * factor;
      view.minY = cy + (view.minY - cy) * factor;
      view.maxY = cy + (view.maxY - cy) * factor;
      draw();
    }}, {{ passive: false }});

    window.addEventListener('resize', resize);
    initView();
    resize();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    targets = [target.strip() for target in args.targets.split(",") if target.strip()]

    model, ckpt = load_low_model(args.low_checkpoint, device)
    target_scaler = TargetScaler.load(args.prepared_dir / "target_scaler.json")

    center = pd.read_parquet(args.prepared_dir / "center_blocks.parquet")
    center = center.loc[center["has_targets"]].reset_index(drop=True)
    north_raw = pd.read_parquet(args.prepared_dir / "north_blocks.parquet").reset_index(drop=True)
    north_ctx = attach_context(north_raw, ckpt, args.prepared_dir, device)

    sequence_length = args.sequence_length or int(ckpt["model_config"]["sequence_length"])
    print(f"Generating north predictions: rows={len(north_ctx)}, sequence_length={sequence_length}")
    north_pred = predict_domain(
        "north_all",
        north_ctx,
        model,
        ckpt,
        target_scaler,
        sequence_length,
        args.batch_size,
        device,
        not args.no_progress,
        args.decode_mode,
        args.softmax_temperature,
    )

    layers = [
        make_layer(
            "center_true",
            "Center/South BM true",
            center,
            targets,
            "",
            args.max_points_per_layer,
            args.seed,
            marker="circle",
            default_visible=True,
        ),
        make_layer(
            "north_pred",
            "North BM generated",
            north_pred,
            targets,
            "pred_",
            args.max_points_per_layer,
            args.seed + 1,
            marker="square",
            default_visible=True,
        ),
    ]
    if not args.no_north_known_true:
        north_known = north_raw.loc[north_raw["has_targets"]].reset_index(drop=True)
        layers.append(
            make_layer(
                "north_true_known",
                "North BM known true",
                north_known,
                targets,
                "",
                args.max_points_per_layer,
                args.seed + 2,
                marker="circle",
                default_visible=False,
            )
        )

    payload = {
        "checkpoint": str(args.low_checkpoint),
        "prepared_dir": str(args.prepared_dir),
        "decode_mode": args.decode_mode,
        "softmax_temperature": args.softmax_temperature,
        "targets": targets,
        "domains": compute_domains(layers, targets),
        "layers": layers,
    }
    args.output_html.write_text(render_html(payload), encoding="utf-8")
    print(f"Saved HTML: {args.output_html}")


if __name__ == "__main__":
    main()
