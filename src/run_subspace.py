#!/usr/bin/env python
"""
RQ-3 — split residual into k sub-spaces.
"""
import argparse, yaml, numpy as np
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt, pathlib
from thesis_utils import (set_seed, load_model, get_prompts,
                          collect_resid, get_W_OV, kl_div, bootstrap_ci, save_json)

def main(cfg):
    cfg.setdefault("dataset", "wikitext2")
    cfg.setdefault("dtype",   "float16")

    set_seed(cfg["seed"])

    model = load_model(cfg["model"], dtype=cfg["dtype"])
    tok   = model.tokenizer


    W_OV  = get_W_OV(model, cfg["layer"]).mean(0)

    prompts = get_prompts(dataset=cfg["dataset"],
                          n_prompts=cfg["n_prompts"],
                          max_len=cfg["max_len"],
                          seed=cfg["seed"])
    d_model = model.cfg.d_model
    widths  = [d_model // cfg["k_sub"]] * cfg["k_sub"]
    widths[-1] += d_model % cfg["k_sub"]
    idx = np.cumsum([0]+widths)

    rec = []
    for i in range(0, len(prompts), cfg["batch"]):
        batch = prompts[i:i+cfg["batch"]]
        device = model.W_U.device
        ids = tok(batch, padding=True, truncation=True,
                  max_length=cfg["max_len"], return_tensors="pt").input_ids.to(device)
        gt, resid = collect_resid(model, ids, None, cfg["layer"], which="mid")

        for s,(lo,hi) in enumerate(zip(idx[:-1], idx[1:])):
            kl_log = kl_div(gt, resid[:,lo:hi] @ model.W_U[lo:hi]).detach().cpu().numpy()
            kl_ov  = kl_div(gt, resid[:,lo:hi] @ W_OV[lo:hi] @ model.W_U).detach().cpu().numpy()
            rec.extend([dict(sub=s, dkl=float(a-b)) for a,b in zip(kl_log, kl_ov)])

    # compute CI per sub‑space
    stats = {s: bootstrap_ci(np.array([r["dkl"] for r in rec if r["sub"]==s]))
             for s in range(cfg["k_sub"])}
    save_json(dict(cfg=cfg, rec=rec, ci=stats), f"{cfg['out_dir']}", "enrich")

    df = pd.DataFrame(rec)
    plt.figure(figsize=(6,4))
    sns.violinplot(data=df, x="sub", y="dkl", inner="quartile", palette="Blues")
    plt.xlabel("Sub-space ID"); plt.ylabel("ΔKL (Logit−OV)")
    plt.title(f"OV advantage per sub-space  (k={cfg['k_sub']})")
    png = pathlib.Path(cfg["out_dir"]) / "enrich_violin.png"
    plt.tight_layout(); plt.savefig(png, dpi=300); print("📈 saved", png)


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/enrich.yaml"))
    main(cfg)
