from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .vector_quantizer import VectorQuantizer


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class TopVQTransformer(nn.Module):
    """Top-level VQ autoencoder for assay/lithology tokens."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        codebook_size: int = 256,
        dropout: float = 0.1,
        max_sequence_length: int = 2048,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Embedding(max_sequence_length, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.quantizer = VectorQuantizer(codebook_size=codebook_size, embedding_dim=d_model)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=max(1, n_layers // 2))
        self.output = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        pos = torch.arange(seq_len, device=x.device).clamp_max(self.max_sequence_length - 1)
        h = self.input_proj(x) + self.pos(pos).unsqueeze(0)
        key_padding_mask = None if mask is None else ~mask
        z_e = self.encoder(h, src_key_padding_mask=key_padding_mask)
        z_q, codes, vq_loss, perplexity = self.quantizer(z_e)
        dec = self.decoder(z_q, src_key_padding_mask=key_padding_mask)
        recon = self.output(dec)
        return {"recon": recon, "codes": codes, "z_q": z_q, "vq_loss": vq_loss, "perplexity": perplexity}

    @torch.no_grad()
    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x, mask=mask)
        return out["codes"], out["z_q"]


class LowVQVAE2(nn.Module):
    """Low-level block VQ-VAE with a causal Transformer code prior."""

    def __init__(
        self,
        block_dim: int,
        target_dim: int = 5,
        top_dim: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        codebook_size: int = 256,
        dropout: float = 0.1,
        max_sequence_length: int = 2048,
    ):
        super().__init__()
        self.block_dim = block_dim
        self.target_dim = target_dim
        self.top_dim = top_dim
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

        cond_dim = block_dim + top_dim
        self.cond_proj = nn.Linear(cond_dim, d_model)
        self.target_encoder = nn.Sequential(
            nn.Linear(cond_dim + target_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.quantizer = VectorQuantizer(codebook_size=codebook_size, embedding_dim=d_model)
        self.decoder = nn.Sequential(
            nn.Linear(cond_dim + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, target_dim),
        )

        self.pos = nn.Embedding(max_sequence_length, d_model)
        prior_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.prior = nn.TransformerEncoder(prior_layer, num_layers=n_layers)
        self.code_logits = nn.Linear(d_model, codebook_size)

    def encode_targets(
        self,
        block_features: torch.Tensor,
        top_context: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = torch.cat([block_features, top_context], dim=-1)
        z_e = self.target_encoder(torch.cat([cond, targets], dim=-1))
        return self.quantizer(z_e)

    def decode(
        self,
        block_features: torch.Tensor,
        top_context: torch.Tensor,
        z_q: torch.Tensor,
    ) -> torch.Tensor:
        cond = torch.cat([block_features, top_context], dim=-1)
        return self.decoder(torch.cat([cond, z_q], dim=-1))

    def prior_logits(
        self,
        block_features: torch.Tensor,
        top_context: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = block_features.shape
        cond = torch.cat([block_features, top_context], dim=-1)
        h = self.cond_proj(cond)
        pos = torch.arange(seq_len, device=h.device).clamp_max(self.max_sequence_length - 1)
        h = h + self.pos(pos).unsqueeze(0)
        key_padding_mask = None if mask is None else ~mask
        h = self.prior(
            h,
            mask=causal_mask(seq_len, h.device),
            src_key_padding_mask=key_padding_mask,
        )
        return self.code_logits(h)

    def forward(
        self,
        block_features: torch.Tensor,
        top_context: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z_q, codes, vq_loss, perplexity = self.encode_targets(block_features, top_context, targets)
        recon = self.decode(block_features, top_context, z_q)
        logits = self.prior_logits(block_features, top_context, mask=mask)
        return {
            "recon": recon,
            "codes": codes,
            "z_q": z_q,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "logits": logits,
        }

    @torch.no_grad()
    def generate(
        self,
        block_features: torch.Tensor,
        top_context: torch.Tensor,
        mask: torch.Tensor | None = None,
        decode_mode: str = "hard",
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.prior_logits(block_features, top_context, mask=mask)
        codes = logits.argmax(dim=-1)
        if decode_mode == "soft":
            temp = max(float(temperature), 1e-6)
            probs = F.softmax(logits / temp, dim=-1)
            z_q = probs @ self.quantizer.embedding.weight
        elif decode_mode == "hard":
            z_q = self.quantizer.embedding(codes)
        else:
            raise ValueError(f"Unknown decode_mode: {decode_mode}")
        pred = self.decode(block_features, top_context, z_q)
        return pred, codes


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = (pred - target).pow(2)
    if mask is None:
        return loss.mean()
    loss = loss * mask.unsqueeze(-1).to(loss.dtype)
    return loss.sum() / (mask.sum().clamp_min(1).to(loss.dtype) * target.size(-1))


def code_ce_loss(logits: torch.Tensor, codes: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), codes.reshape(-1), reduction="none")
    if mask is None:
        return loss.mean()
    loss = loss.view_as(codes) * mask.to(loss.dtype)
    return loss.sum() / mask.sum().clamp_min(1).to(loss.dtype)


def _flatten_valid(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        return values[mask]
    return values.reshape(-1, values.size(-1))


def correlation_matrix(values: torch.Tensor) -> torch.Tensor:
    if values.size(0) < values.size(1) + 2:
        return torch.eye(values.size(1), device=values.device, dtype=values.dtype)
    values = values - values.mean(dim=0, keepdim=True)
    values = values / values.std(dim=0, keepdim=True).clamp_min(1e-6)
    return values.t() @ values / max(1, values.size(0) - 1)


def correlation_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    pred = _flatten_valid(pred, mask)
    target = _flatten_valid(target, mask)
    if pred.size(0) < pred.size(1) + 2:
        return pred.new_tensor(0.0)
    pred_corr = correlation_matrix(pred)
    target_corr = correlation_matrix(target)
    return F.mse_loss(pred_corr, target_corr)


def reference_correlation_loss(
    pred: torch.Tensor,
    reference_corr: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = _flatten_valid(pred, mask)
    if pred.size(0) < pred.size(1) + 2:
        return pred.new_tensor(0.0)
    pred_corr = correlation_matrix(pred)
    reference_corr = reference_corr.to(device=pred.device, dtype=pred.dtype)
    return F.mse_loss(pred_corr, reference_corr)


class TopPriorTransformer(nn.Module):
    """Autoregressive prior for block-level top codes.

    During training the labels come from the known block-to-assay
    correspondence. During inference the model generates top codes from block
    features and previously generated top codes.
    """

    def __init__(
        self,
        block_dim: int,
        codebook_size: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_sequence_length: int = 2048,
    ):
        super().__init__()
        self.block_dim = block_dim
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.block_proj = nn.Linear(block_dim, d_model)
        self.prev_code = nn.Embedding(codebook_size + 1, d_model)
        self.bos_code = codebook_size
        self.pos = nn.Embedding(max_sequence_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, codebook_size)

    def forward(
        self,
        block_features: torch.Tensor,
        prev_codes: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = block_features.shape
        pos = torch.arange(seq_len, device=block_features.device).clamp_max(self.max_sequence_length - 1)
        h = self.block_proj(block_features) + self.prev_code(prev_codes) + self.pos(pos).unsqueeze(0)
        key_padding_mask = None if mask is None else ~mask
        h = self.transformer(
            h,
            mask=causal_mask(seq_len, h.device),
            src_key_padding_mask=key_padding_mask,
        )
        return self.output(h)

    @torch.no_grad()
    def generate(self, block_features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, _ = block_features.shape
        prev = torch.full((bsz, seq_len), self.bos_code, dtype=torch.long, device=block_features.device)
        codes = torch.zeros((bsz, seq_len), dtype=torch.long, device=block_features.device)
        for pos in range(seq_len):
            logits = self.forward(block_features[:, : pos + 1], prev[:, : pos + 1], None if mask is None else mask[:, : pos + 1])
            code = logits[:, -1].argmax(dim=-1)
            codes[:, pos] = code
            if pos + 1 < seq_len:
                prev[:, pos + 1] = code
        return codes

    @torch.no_grad()
    def generate_with_prefix(
        self,
        block_features: torch.Tensor,
        prefix_codes: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = block_features.shape
        prefix_len = int(prefix_codes.size(1))
        if prefix_len >= seq_len:
            return prefix_codes[:, :seq_len]

        prev = torch.full((bsz, seq_len), self.bos_code, dtype=torch.long, device=block_features.device)
        codes = torch.zeros((bsz, seq_len), dtype=torch.long, device=block_features.device)
        if prefix_len > 0:
            codes[:, :prefix_len] = prefix_codes
            if prefix_len > 1:
                prev[:, 1:prefix_len] = prefix_codes[:, :-1]
            prev[:, prefix_len] = prefix_codes[:, -1]

        for pos in range(prefix_len, seq_len):
            logits = self.forward(
                block_features[:, : pos + 1],
                prev[:, : pos + 1],
                None if mask is None else mask[:, : pos + 1],
            )
            code = logits[:, -1].argmax(dim=-1)
            codes[:, pos] = code
            if pos + 1 < seq_len:
                prev[:, pos + 1] = code
        return codes


def shift_codes_right(codes: torch.Tensor, bos_code: int) -> torch.Tensor:
    prev = torch.full_like(codes, bos_code)
    prev[:, 1:] = codes[:, :-1]
    return prev
