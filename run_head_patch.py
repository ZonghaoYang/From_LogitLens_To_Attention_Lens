"""
RQ‑2.2  —  Head‑OV causal patching
"""

from __future__ import annotations
import argparse, yaml, json, time, pathlib
import numpy as np, torch, matplotlib.pyplot as plt, seaborn as sns
from transformer_lens import utils as tl_utils
from thesis_utils     import (set_seed, load_model, get_prompts,
                              get_W_OV, kl_div, save_json)

# ────────────────── lightweight resid extractor ──────────────────────────
def get_resid_mid(model, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
    """
    Return residual stream at layer *output* (post‑MLP); if model does not
    expose resid_mid, gracefully fall back to resid_post.  Shape (B,d).
    """
    import transformer_lens.utils as tl_utils

    # prefer resid_mid, else resid_post
    for which in ("resid_mid", "resid_post"):
        hook_name = tl_utils.get_act_name(which, layer)
        if hook_name in model.mod_dict:
            break

    buf = {}
    def grab(t, hook): buf["resid"] = t[:, -1, :].detach()

    with torch.no_grad():
        _ = model.run_with_hooks(input_ids, fwd_hooks=[(hook_name, grab)])
    return buf["resid"]


# ────────────────── head‑specific σ estimator (tiny sample) ───────────────
def est_sigma(model, layer, head, sample_ids):
    resid = get_resid_mid(model, sample_ids, layer)
    W = get_W_OV(model, layer, [head])[0].to(resid)
    vec = resid @ W
    return vec.std().item()

# ────────────────── main routine ──────────────────────────────────────────
def main(cfg):
    set_seed(cfg["seed"])
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    model   = load_model(cfg["model"], dtype=cfg.get("dtype","float16"),
                         device=device)
    tok     = model.tokenizer
    W_U     = model.W_U

    heat = json.load(open(cfg["heatmap_json"]))
    dkl  = np.array(heat["dkl"])
    sel  = np.argwhere(dkl >= cfg["dkl_thr"])

    prompts = get_prompts(dataset=cfg["dataset"],
                          n_prompts=cfg["n_prompts"],
                          max_len=cfg["max_len"],
                          seed=cfg["seed"])

    BATCH = cfg["batch"]
    LAMBDAS = cfg["lambdas"]

    rec = {}
    for L, H in sel:
        W_h = get_W_OV(model, L, [int(H)])[0].to(W_U.dtype).to(device)

        sample_ids = tok(prompts[:64], padding=True, truncation=True,
                         max_length=cfg["max_len"],
                         return_tensors="pt").input_ids.to(device)
        sigma = est_sigma(model, int(L), int(H), sample_ids)
        torch.cuda.empty_cache()

        kl_base_all, patch_dict = [], {lam:[] for lam in LAMBDAS}

        idx = 0
        while idx < len(prompts):
            sub = prompts[idx: idx+BATCH]; idx += BATCH
            try:
                ids = tok(sub, padding=True, truncation=True,
                          max_length=cfg["max_len"],
                          return_tensors="pt").input_ids.to(device)

                resid = get_resid_mid(model, ids, int(L))
                gt    = torch.log_softmax(resid @ W_U, -1)
                kl_base_all.extend(
                    kl_div(gt.exp(), resid @ W_U).cpu().tolist())

                vec_h = resid @ W_h

                for lam in LAMBDAS:
                    resid_mod = resid + (lam * sigma) * vec_h
                    kl_mod = kl_div(gt.exp(), resid_mod @ W_U)
                    patch_dict[lam].extend(kl_mod.cpu().tolist())

                del resid, ids; torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if BATCH == 1:
                    raise RuntimeError("Even batch=1 OOM, reduce max_len")
                BATCH //= 2
                print(f"[OOM] batch→{BATCH}; retry current chunk")
                idx -= BATCH

        rec[f"L{L}-H{H}"] = dict(
            sigma=sigma, kl_base=kl_base_all, kl_patch=patch_dict)


    out_dir = pathlib.Path(cfg["out_dir"]); out_dir.mkdir(exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")
    js  = save_json({"cfg": cfg, "rec": rec}, out_dir, f"head_patch_{tag}")
    print("✅  saved", js.name)


    for key, blob in rec.items():
        μ_base = np.mean(blob["kl_base"])
        xs, ys = [], []
        for lam in LAMBDAS:
            xs.append(lam)
            ys.append(μ_base - np.mean(blob["kl_patch"][lam]))
        plt.plot(xs, ys, "-o", label=key)
    plt.axhline(0, ls="--", c="gray")
    plt.xlabel("λ · σ"); plt.ylabel("ΔKL gain (nats)")
    plt.title(f"{cfg['model']}  OV patch effect")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(out_dir / "head_patch_overview.png", dpi=300); plt.close()

# ─── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    main(cfg)
