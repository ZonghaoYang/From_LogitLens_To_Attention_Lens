"""
RQ-1 — single layer·head：Logit-Lens vs OV-Lens KL & Top-k
"""
import argparse, yaml, numpy as np
import matplotlib.pyplot as plt, pathlib
from thesis_utils import (set_seed, load_model, get_prompts, collect_resid, get_W_OV, kl_div, topk_overlap, entropy, bootstrap_ci, save_json)
from scipy import stats

def main(cfg):
    cfg.setdefault("dataset", "wikitext2")
    cfg.setdefault("dtype",   "float16")

    set_seed(cfg["seed"])
    model = load_model(cfg["model"], dtype=cfg["dtype"])

    tok   = model.tokenizer

    W_U   = model.W_U
    W_OV  = get_W_OV(model, cfg["layer"], [cfg["head"]])[0]

    prompts = get_prompts(dataset=cfg["dataset"],
                          n_prompts=cfg["n_prompts"],
                          max_len=cfg["max_len"],
                          seed=cfg["seed"])

    rec = {"kl_logit": [], "kl_ov": [],
           "ent_logit": [], "ent_ov": [],
           "top1_logit": [], "top1_ov": []}

    for i in range(0, len(prompts), cfg["batch"]):
        batch = prompts[i : i+cfg["batch"]]
        device = model.W_U.device
        ids = tok(batch, padding=True, truncation=True,
                  max_length=cfg["max_len"], return_tensors="pt").input_ids.to(device)
        gt, resid = collect_resid(model, ids, None, cfg["layer"], which="mid")
        l_logit = resid @ W_U
        l_ov    = resid @ W_OV @ W_U

        rec["kl_logit"].extend(kl_div(gt, l_logit).cpu().tolist())
        rec["kl_ov"].extend(kl_div(gt, l_ov).cpu().tolist())
        rec["ent_logit"].extend(entropy(l_logit).cpu().tolist())
        rec["ent_ov"].extend(entropy(l_ov).cpu().tolist())
        rec["top1_logit"].extend(topk_overlap(gt, l_logit, 1).cpu().tolist())
        rec["top1_ov"].extend(topk_overlap(gt, l_ov, 1).cpu().tolist())
    ci = {k: bootstrap_ci(np.array(v)) for k, v in rec.items()}
    rec["ci"] = {k: [float(a), float(b)] for k, (a, b) in ci.items()}

    t_stat, p_val = stats.ttest_rel(rec["kl_logit"], rec["kl_ov"])
    rec["t_test"] = {"t": float(t_stat), "p": float(p_val)}

    save_json(dict(cfg=cfg, rec=rec), f"{cfg['out_dir']}", "rq1")

    out_dir = pathlib.Path(cfg["out_dir"])
    kl_log  = np.array(rec["kl_logit"])
    kl_ov   = np.array(rec["kl_ov"])

    plt.figure(figsize=(4.2, 4))
    b = plt.boxplot([kl_log, kl_ov], labels=["Logit‑Lens", "OV‑Lens"],
                    patch_artist=True, widths=.55)
    for patch, col in zip(b['boxes'], ["#CCCCCC", "#8FAADC"]):
        patch.set_facecolor(col)

    for i, data in enumerate([kl_log, kl_ov], start=1):
        plt.scatter(i, data.mean(), marker="D", color="black", zorder=3, s=25)

    plt.ylabel("KL divergence (nats) ↓")
    plt.title(f"Layer {cfg['layer']}·Head {cfg['head']}  "
              f"(t={t_stat:.2f}, p={p_val:.1e})")
    plt.tight_layout()
    png = out_dir / f"rq1_box_L{cfg['layer']}H{cfg['head']}.png"
    plt.savefig(png, dpi=300); plt.close()
    print("📈 saved", png)


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/rq1.yaml"))
    main(cfg)
