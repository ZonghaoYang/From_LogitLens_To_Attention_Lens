
from __future__ import annotations
import argparse, yaml, pathlib, time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from thesis_utils import (
    set_seed, load_model, get_prompts,
    collect_resid, get_W_OV, kl_div, save_json
)

plt.rcParams.update({"figure.facecolor": "white",
                     "font.size": 11,
                     "axes.labelpad": 4})

# ---------------------------------------------------------------------------
def main(cfg: dict):
    cfg.setdefault("dataset", "wikitext2")
    cfg.setdefault("dtype",   "float16")
    cfg.setdefault("annot_thr", 0.5)
    cfg.setdefault("fig_scale", 0.45)

    set_seed(cfg["seed"])
    model = load_model(cfg["model"], dtype=cfg["dtype"])
    tok   = model.tokenizer
    prompts = get_prompts(dataset=cfg["dataset"],
                          n_prompts=cfg["n_prompts"],
                          max_len=cfg["max_len"],
                          seed=cfg["seed"])

    L, H  = model.cfg.n_layers, model.cfg.n_heads
    dkl   = np.zeros((L, H), dtype=np.float32)

    for ℓ in range(L):
        W_OV_L = get_W_OV(model, ℓ)
        for h in range(H):
            kl_log = kl_ov = 0.0
            for i in range(0, len(prompts), cfg["batch"]):
                batch  = prompts[i:i+cfg["batch"]]
                ids    = tok(batch, padding=True, truncation=True,
                             max_length=cfg["max_len"],
                             return_tensors="pt").input_ids.to(model.W_U.device)
                gt, resid = collect_resid(model, ids, None, ℓ, which="mid")
                kl_log += kl_div(gt, resid @ model.W_U).sum().item()
                kl_ov  += kl_div(gt, resid @ W_OV_L[h] @ model.W_U).sum().item()
            dkl[ℓ, h] = (kl_log - kl_ov) / len(prompts)
        print(f"layer {ℓ:02d} done")


    tag = time.strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(cfg["out_dir"]); out_dir.mkdir(exist_ok=True)
    save_json(dict(cfg=cfg, dkl=dkl.tolist()), out_dir, f"heatmap")

    fig_h = cfg["fig_scale"] * L + 1.2
    fig_w = cfg["fig_scale"] * H + 1.0
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(dkl,
                     cmap="coolwarm", center=0,
                     cbar_kws=dict(label="ΔKL (Logit−OV)", shrink=0.8),
                     annot=False, linewidths=.3, linecolor='gray',
                     xticklabels=range(H), yticklabels=range(L))

    thr = cfg["annot_thr"]
    for i in range(L):
        for j in range(H):
            val = dkl[i, j]
            if abs(val) >= thr:
                ax.text(j + 0.5, i + 0.5,
                        f"{val:.1f}", ha='center', va='center',
                        color='black' if abs(val) < 4 else 'white',
                        fontsize=8)

    plt.xlabel("Head"); plt.ylabel("Layer")
    plt.title(f"{cfg['model']}  ·  ΔKL heat‑map  (n={len(prompts)})")
    plt.tight_layout()

    png = out_dir / f"heatmap_{cfg['model']}_n{len(prompts)}.png"
    plt.savefig(png, dpi=300)
    plt.close()
    print("📈  saved", png)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/heatmap.yaml"))
    main(cfg)
