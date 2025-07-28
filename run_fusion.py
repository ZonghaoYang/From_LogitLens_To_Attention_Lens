#!/usr/bin/env python
"""
RQ‑4 — multi‑head OV‑Lens fusion curve
"""
from __future__ import annotations
import argparse, yaml, pathlib, time, json
import numpy as np, torch, matplotlib.pyplot as plt
from thesis_utils import (
    set_seed, load_model, get_prompts,
    collect_resid, get_W_OV, kl_div,
    fit_mixed_projection, save_json
)

def fuse_matrix(W_OV_H: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return torch.einsum("h,hdn->dn", alpha.to(W_OV_H), W_OV_H)

def run_curve(cfg: dict, seed: int):
    set_seed(seed)
    model  = load_model(cfg["model"], dtype=cfg["dtype"])
    tok    = model.tokenizer; device = model.W_U.device
    heads  = cfg["heads"];  H = len(heads)
    W_OV_H = get_W_OV(model, cfg["layer"], heads).to(device)

    prompts = get_prompts(dataset=cfg["dataset"],
                          n_prompts=cfg["n_prompts"],
                          max_len=cfg["max_len"],
                          seed=seed)
    calib   = prompts[: cfg.get("calib_size", 256)]

    curve, qcases, alpha_hist = [], [], []

    # init
    alpha = torch.zeros(H, device=device); W_fuse = W_OV_H[0].clone()

    for k in range(1, H + 1):
        if cfg["weight"] == "uniform":
            alpha[:k] = 1.0 / k
            W_fuse = fuse_matrix(W_OV_H[:k], alpha[:k]).to(model.W_U.dtype)

        elif cfg["weight"] == "attn":
            if k == 1: alpha.zero_()
            with torch.no_grad():
                dummy = tok("Hello", return_tensors="pt").input_ids.to(device)
                _, cache = model.run_with_cache(dummy)
                att = cache["attn", cfg["layer"]][0, heads[k-1]].mean()
            alpha[k-1] = att.item()
            W_fuse = fuse_matrix(W_OV_H[:k], alpha[:k]/alpha[:k].sum()).to(model.W_U.dtype)

        else:
            try:
                alpha_cpu, W_fuse = fit_mixed_projection(
                    model, cfg["layer"], heads[:k], calib, tok,
                    method=cfg["weight"], calib_size=len(calib))
            except Exception as e:
                print(f"[warn] α solve failed at k={k}: {e}. fallback uniform")
                alpha_cpu = np.ones(k) / k
                W_fuse = fuse_matrix(W_OV_H[:k],
                                     torch.as_tensor(alpha_cpu, device=device))
            alpha.zero_(); alpha[:k] = torch.as_tensor(alpha_cpu, device=device)

        alpha_hist.append(alpha.cpu().tolist())

        # ----- metric -------------------------------------------------------
        kl_log = kl_fuse = 0.0
        for i in range(0, len(prompts), cfg["batch"]):
            batch = prompts[i:i+cfg["batch"]]
            ids = tok(batch, padding=True, truncation=True,
                      max_length=cfg["max_len"],
                      return_tensors="pt").input_ids.to(device)
            gt, resid = collect_resid(model, ids, None, cfg["layer"], which="mid")
            kl_log  += kl_div(gt, resid @ model.W_U).sum().item()
            kl_fuse += kl_div(gt, resid @ W_fuse @ model.W_U).sum().item()

        n = len(prompts)
        curve.append((kl_log/n, kl_fuse/n))
        print(f"seed={seed} k={k} ΔKL={kl_log/n - kl_fuse/n:.4f}")

        if k == H and len(qcases) < 2:
            top0 = torch.topk(resid @ model.W_U, 5, -1).indices[0].cpu()
            top1 = torch.topk(resid @ W_fuse @ model.W_U, 5, -1).indices[0].cpu()
            qcases.append(dict(prompt=batch[0],
                               top_logit=top0.tolist(),
                               top_fuse =top1.tolist()))
    return curve, qcases, alpha_hist

# -------------------------------------------------------------------------
def main(cfg: dict):
    cfg.setdefault("dataset", "wikitext2")
    cfg.setdefault("dtype",   "float16")
    seed_list = cfg.get("seeds", [cfg.get("seed", 0)])

    curves_all, q_all, alpha_all = [], [], []
    for s in seed_list:
        c, q, a = run_curve(cfg, s)
        curves_all.append(c); q_all += q; alpha_all.append(a)

    out_dir = pathlib.Path(cfg["out_dir"]); out_dir.mkdir(exist_ok=True)
    js = save_json(dict(cfg=cfg,
                        curves=curves_all,
                        alpha =alpha_all,
                        qualitative=q_all),
                   out_dir, f"fusion_{cfg['weight']}")
    print("JSON saved:", js.name)

    arr = np.array(curves_all)
    gain = arr[:,:,0].mean(0) - arr[:,:,1].mean(0)
    std  = arr[:,:,1].std(0)
    x = np.arange(1, len(gain)+1)

    plt.figure(figsize=(6,4))
    plt.plot(x, gain, "-o"); plt.fill_between(x, gain-std, gain+std, alpha=.25)
    plt.ylabel("ΔKL gain (nats)"); plt.xlabel("#Heads fused"); plt.grid()
    plt.title(f"{cfg['model']} · L{cfg['layer']} · {cfg['weight'].upper()}")
    out_png = out_dir / f"fusion_{cfg['model']}_{cfg['weight']}_agg.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
    print("PNG saved:", out_png.name)

if __name__ == "__main__":
    pa = argparse.ArgumentParser(); pa.add_argument("--config", required=True)
    args, unk = pa.parse_known_args()

    cfg = yaml.safe_load(open(args.config))
    if "--override" in unk:
        for kv in unk[unk.index("--override")+1:]:
            k,v = kv.split("=",1); cfg[k] = yaml.safe_load(v)
    main(cfg)
