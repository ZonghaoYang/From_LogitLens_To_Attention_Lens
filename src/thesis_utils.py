from __future__ import annotations

import json, logging, os, random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer, utils

__all__ = [
    "THESIS_DIR", "setup_paths", "set_seed",
    "load_model", "get_prompts", "kl_div", "topk_overlap",
    "get_W_OV", "collect_resid", "save_json", "bootstrap_ci",
    "get_headwise_logits", "fit_mixed_projection",
]


THESIS_DIR: str = "/content/drive/MyDrive/From_LogitLens_To_AttentionLens"
HF_CACHE_DIR = "llm_cache/hf"
DS_CACHE_DIR = "llm_cache/datasets"


def setup_paths(thesis_dir: Optional[str] = None) -> Path:
    """Create project folders, set HF env vars, add <src> to PYTHONPATH."""
    global THESIS_DIR
    if thesis_dir is not None:
        THESIS_DIR = thesis_dir
    root = Path(THESIS_DIR).expanduser(); root.mkdir(parents=True, exist_ok=True)

    for sub in ("src", HF_CACHE_DIR, DS_CACHE_DIR):
        (root / sub).mkdir(parents=True, exist_ok=True)

    os.environ.update({
        "HF_HOME": str(root / HF_CACHE_DIR),
        "TRANSFORMERS_CACHE": str(root / HF_CACHE_DIR),
        "HF_DATASETS_CACHE": str(root / DS_CACHE_DIR),
    })
    import sys
    src_path = str(root / "src")
    if src_path not in sys.path:
        sys.path.append(src_path)

    logging.basicConfig(level=logging.INFO)
    logging.info("Project folders ready at %s", root)
    return root

# ---------------------------------------------------------------------------
# Reproducibility

def set_seed(seed: int) -> None:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

# ---------------------------------------------------------------------------
# Model & data helpers

def load_model(model_name: str,
               *,
               device: Optional[str] = None,
               dtype: torch.dtype | str = torch.float16,
               seed: Optional[int] = None) -> HookedTransformer:
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if seed is not None:
        set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return HookedTransformer.from_pretrained(
        model_name,
        device=device,
        torch_dtype=dtype,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        n_devices=1,
    )


# ─── thesis_utils.py ──────────────────────────────────────────────────────────
from datasets import load_dataset, load_dataset_builder

def get_prompts(
    *,
    dataset: str = "wikitext2",
    split: str = "validation",
    n_prompts: int | None = None,
    min_len: int = 20,
    max_len: int | None = None,
    seed: int = 0,
    join_lines: bool = False ,
) -> List[str]:
    """Return a list of text prompts for the chosen dataset."""
    set_seed(seed)

    if dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    elif dataset == "wikitext103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        if join_lines:
            merged = []
            buf = []
            for rec in ds:
                txt = rec["text"].strip()
                if txt == "":
                    if buf:
                        merged.append({"text": " ".join(buf)})
                        buf = []
                else:
                    buf.append(txt)
            ds = merged

    elif dataset == "wikitext103_sub5k":
        ds_iter = load_dataset("wikitext", "wikitext-103-raw-v1",
                               split="validation", streaming=True)
        ds = []
        for r in ds_iter:
            txt = r["text"].strip()
            if len(txt) > min_len:
                ds.append({"text": txt})
            if len(ds) >= 5_000:
                break

    elif dataset == "trex":
        ds_raw = load_dataset("lama", "trex", split="train",
                              trust_remote_code=True)

        def extract_text(rec):
            txt = (
                rec.get("masked_sentence") or
                rec.get("masked_sentences") or
                rec.get("template") or
                rec.get("query")
            )
            if txt is None:
                return None
            return txt.replace("[MASK]", rec["obj_label"]).strip()

        ds = [{"text": t} for rec in ds_raw if (t := extract_text(rec))]



    elif dataset == "counterfact":
        ds = load_dataset("counterfact")["test"]
        ds = [{"text": r["prompt"].split("[SEP]")[0]} for r in ds]

    else:
        raise ValueError(f"Unknown dataset={dataset}")

    seqs = [r["text"].strip() for r in ds if len(r["text"].strip()) > min_len]
    random.shuffle(seqs)

    if n_prompts is not None:
        seqs = seqs[:n_prompts]
    if max_len is not None:
        seqs = [s[: max_len * 2] for s in seqs]
    return seqs

# ---------------------------------------------------------------------------
# Maths helpers

def kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = torch.log_softmax(p_logits, -1); q = torch.log_softmax(q_logits, -1)
    return torch.sum(torch.exp(p) * (p - q), dim=-1)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits.float(), -1)
    return -(p * torch.log(p + 1e-9)).sum(-1)


def topk_overlap(gt: torch.Tensor, pred: torch.Tensor, k: int = 5) -> torch.Tensor:
    return (torch.topk(gt, k).indices == torch.topk(pred, k).indices).any(-1)

# ---------------------------------------------------------------------------
# OV matrix cache

_HOOK_MAP = {"pre": "resid_pre", "mid": "resid_mid", "post": "resid_post"}
_OV_CACHE_KEY = "_cached_W_OV"

def _compute_full_OV(model: HookedTransformer) -> torch.Tensor:
    if hasattr(model.cfg, _OV_CACHE_KEY):
        return getattr(model.cfg, _OV_CACHE_KEY)
    L, H, d = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_model
    W_OV = torch.empty((L, H, d, d), dtype=model.W_U.dtype, device=model.W_U.device)
    for ℓ in range(L):
        W_V = model.blocks[ℓ].attn.W_V
        W_O = model.blocks[ℓ].attn.W_O
        W_OV[ℓ] = torch.einsum("hmn,hdm->hdn", W_O, W_V)
        if ℓ == 0 and torch.rand(()) < 1e-2:
          h_rand = torch.randn(d, device=W_V.device)
          assert torch.allclose(W_O @ (W_V @ h_rand), W_OV[ℓ][0] @ h_rand, atol=1e-5), \
              "OV direction mismatch"
    setattr(model.cfg, _OV_CACHE_KEY, W_OV)
    return W_OV


def get_W_OV(model: HookedTransformer, layer: int, heads: Optional[Sequence[int]] = None) -> torch.Tensor:
    W = _compute_full_OV(model)[layer]
    return W if heads is None else W[torch.tensor(heads, device=W.device)]

# ---------------------------------------------------------------------------
# Residual extraction

def collect_resid(model: HookedTransformer, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor],
                  layer: int, *, which: str = "mid", pos: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    hook_str = utils.get_act_name(_HOOK_MAP.get(which, "resid_post"), layer)
    with torch.no_grad():
        logits, cache = model.run_with_cache(
            input_ids, attention_mask=attention_mask,
            names_filter=lambda n: n == hook_str,
        )
    if hook_str not in cache:
        with torch.no_grad():
            logits, cache = model.run_with_cache(input_ids, attention_mask=attention_mask)
        hook_str = utils.get_act_name("resid_post", layer)
    return logits[:, pos, :], cache[hook_str][:, pos, :]

# ---------------------------------------------------------------------------
# I/O helper

def save_json(obj: Dict, out_dir: Path | str, prefix: str) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"{prefix}_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
    with open(fn, "w") as f:
        json.dump(obj, f, indent=2)
    logging.info("Saved %s", fn.name)
    return fn

# ---------------------------------------------------------------------------
# Statistics helper

def bootstrap_ci(arr: np.ndarray, *, n: int = 1000, ci: float = 0.95, agg=np.mean) -> Tuple[float, float]:
    stats = [agg(np.random.choice(arr, size=arr.shape[0], replace=True)) for _ in range(n)]
    lo, hi = np.percentile(stats, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)

# ---------------------------------------------------------------------------
# Patchscopes utilities

def get_headwise_logits(resid: torch.Tensor, W_OV_heads: torch.Tensor, W_U: torch.Tensor) -> torch.Tensor:
    proj = torch.einsum("bd,hdn->bhn", resid, W_OV_heads)
    return torch.einsum("bhn,nv->bhv", proj, W_U)


# ─── thesis_utils.py ────────────────────────────────
def fit_mixed_projection(
    model: HookedTransformer,
    layer: int,
    heads: Sequence[int],
    prompts: Sequence[str],
    tok,
    *,
    method: str = "nnls",
    calib_size: int = 256,
    which: str = "mid",
    device: Optional[str] = None,
) -> Tuple[np.ndarray, torch.Tensor]:
    device = device or model.W_U.device
    dtype  = model.W_U.dtype

    W_OV_H = get_W_OV(model, layer, heads).to(device)
    H      = len(heads)

    # ---- 1. build calibration matrices ------------------------------------
    ids = tok(prompts[:calib_size], padding=True, truncation=True,
              max_length=model.cfg.n_ctx, return_tensors="pt").input_ids.to(device)
    gt, resid = collect_resid(model, ids, None, layer, which=which)

    head_logits = torch.einsum("bd,hdn->bhn", resid, W_OV_H)
    head_logits = torch.einsum("bhn,nv->bhv", head_logits, model.W_U)
    gt_logits   = resid @ model.W_U

    B, V = gt_logits.shape
    X = head_logits.permute(0, 2, 1).reshape(B * V, H).float().cpu()
    y = gt_logits.reshape(B * V, 1).float().cpu()

    # ---- 2. solve α --------------------------------------------------------
    if method == "nnls":
        with torch.no_grad():
            sol, *_ = torch.linalg.lstsq(X, y, driver="gels")
        alpha = sol.squeeze(1).clamp(min=0).detach().numpy()

    elif method == "greedy":
        remaining, selected = set(range(H)), []
        current = torch.zeros_like(gt_logits)
        while remaining:
            best, best_kl = None, 1e9
            for h in remaining:
                cand = current + head_logits[:, h, :]
                kl = kl_div(gt_logits, cand).mean().item()
                if kl < best_kl:
                    best, best_kl = h, kl
            selected.append(best); remaining.remove(best)
            current += head_logits[:, best, :]
        alpha = np.zeros(H); alpha[selected] = 1 / len(selected)

    else:
        raise ValueError("method must be 'nnls' or 'greedy'")

    if alpha.sum() == 0:
        alpha[:] = 1.0 / H
    alpha = alpha / alpha.sum()

    alpha_t = torch.as_tensor(alpha, device=device, dtype=W_OV_H.dtype)
    W_mix = torch.einsum("h,hdn->dn", alpha_t, W_OV_H).to(dtype)
    return alpha, W_mix


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    setup_paths()
    set_seed(0)
    m = load_model("pythia-70m", device="cpu", dtype=torch.float32)
    tok = m.tokenizer
    prompts = get_prompts(n_prompts=32)
    a, _ = fit_mixed_projection(m, 0, list(range(8)), prompts, tok, method="nnls")
    print("alpha:", a.round(3), "Σ=", a.sum())
