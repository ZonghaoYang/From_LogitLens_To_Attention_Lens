#!/usr/bin/env python

import argparse, json, pathlib, time, yaml, glob, re
import numpy as np, torch, matplotlib.pyplot as plt, seaborn as sns
from transformer_lens import utils
from thesis_utils import (
    set_seed, load_model, get_prompts,
    collect_resid, get_W_OV, save_json
)

# ---------- Helper ---------------------------------------------------------

def latest_nnls_json(results_dir: pathlib.Path) -> pathlib.Path:
    files = sorted(results_dir.glob("fusion_nnls_20250627_021927.json"))
    if not files:
        raise FileNotFoundError("fusion_nnls_*.json not found. Please run RQ-4-NNLS first.")
    return files[-1]


def compute_obsprop_heads(model, layer: int, input_ids, heads):
    hook_v = utils.get_act_name("v", layer)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_ids, names_filter=lambda n: n == hook_v
        )
    v = cache[hook_v][:, -1, heads, :]
    W_O = model.blocks[layer].attn.W_O[heads].to(v)
    return torch.einsum("bhd,hdm->bm", v, W_O)

# ---------- Main -----------------------------------------------------------

def main(cfg):
    set_seed(cfg["seed"])

    model = load_model(cfg["model"], dtype=cfg.get("dtype","float16"))
    tok   = model.tokenizer
    device= model.W_U.device
    prompts = get_prompts(
        dataset = cfg.get("dataset","wikitext2"),
        n_prompts = cfg["n_prompts"],
        max_len   = cfg["max_len"],
        seed      = cfg["seed"]
    )

    layer = cfg["layer"]


    results_dir = pathlib.Path(cfg["out_dir"])
    nnls_path   = latest_nnls_json(results_dir)
    with open(nnls_path) as f:
        fuse_blob = json.load(f)
    heads_full  = fuse_blob["cfg"]["heads"]
    alpha_full  = np.array(fuse_blob["alpha"][0][-1])
    k = 3
    top_idx     = np.argsort(-alpha_full)[:k]
    sel_heads   = [heads_full[i] for i in top_idx]
    alpha       = alpha_full[top_idx]

    # -------------------------------------------
    W_OV_H = get_W_OV(model, layer, sel_heads).to(device)
    alpha_t= torch.as_tensor(alpha, device=device, dtype=W_OV_H.dtype)
    W_fuse = torch.einsum("h,hdn->dn", alpha_t, W_OV_H)

    # -------------------------------------------------------------
    cos_all = []
    for i in range(0, len(prompts), cfg["batch"]):
        batch = prompts[i:i+cfg["batch"]]
        ids   = tok(batch, padding=True, truncation=True,
                    max_length=cfg["max_len"], return_tensors="pt").input_ids.to(device)

        _, resid = collect_resid(model, ids, None, layer, which="mid")
        ov_vec   = resid @ W_fuse

        obs_vec  = compute_obsprop_heads(model, layer, ids, sel_heads)

        cos = torch.nn.functional.cosine_similarity(ov_vec, obs_vec, dim=-1)
        cos_all.extend(cos.cpu().tolist())

    # --------------------------------------------------------
    out = dict(
        cfg   = cfg,
        heads = sel_heads,
        alpha = alpha.tolist(),
        cos   = cos_all,
        mean  = float(np.mean(cos_all)),
        std   = float(np.std(cos_all)),
    )
    tag = time.strftime("%Y%m%d_%H%M%S")
    json_path = save_json(out, results_dir, f"obsprop_k3")

    plt.figure(figsize=(5,4))
    sns.histplot(cos_all, bins=30, kde=True, color="#4C72B0")
    plt.xlabel("Cosine similarity (OV ‖ ObsProp)")
    title = f"{cfg['model']} · L{layer} · heads={sel_heads}"
    plt.title(title)
    txt = f"μ={out['mean']:.3f}  σ={out['std']:.3f}"
    plt.annotate(txt, xy=(0.65,0.9), xycoords="axes fraction")
    png = json_path.with_suffix(".png")
    plt.tight_layout(); plt.savefig(png, dpi=300); plt.close()
    print("✅  saved", json_path.name, "&", png.name)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
