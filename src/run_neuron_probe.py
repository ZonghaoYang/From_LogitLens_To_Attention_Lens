#（RQ1.2）
import argparse, yaml, pathlib, time, json
import torch, numpy as np, matplotlib.pyplot as plt
from transformer_lens import utils as tl_utils
from thesis_utils     import load_model, get_prompts, kl_div, save_json
import matplotlib as mpl
mpl.rcParams.update({"font.size": 11, "axes.titlesize": 13})

# ---------------------------------------------------------------------
def probe(cfg):
    model = load_model(cfg["model"], dtype="bfloat16")
    tok   = model.tokenizer
    dev   = model.W_U.device
    prompts = get_prompts(dataset   = cfg.get("dataset", "wikitext2"),
                          n_prompts = cfg["n_prompts"],
                          seed      = cfg["seed"])
    BATCH  = cfg.get("batch_size", 8)

    hook_names = [tl_utils.get_act_name("mlp_post", L) for L in cfg["layer_ids"]]
    key_list   = [f"L{L}-N{n}" for L, ns in zip(cfg["layer_ids"], cfg["neuron_ids"]) for n in ns]
    kl_buf     = {k: [] for k in key_list}

    for start in range(0, len(prompts), BATCH):
        ids = tok(prompts[start:start+BATCH],
                  padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to(dev).input_ids


        with torch.no_grad():
            logits_full = model(ids)
        gt_last = logits_full[:, -1, :].float().cpu()

        with torch.no_grad():
            _, cache = model.run_with_cache(ids,
                         names_filter=lambda n: n in hook_names)

        for L, n_list in zip(cfg["layer_ids"], cfg["neuron_ids"]):
            hook = tl_utils.get_act_name("mlp_post", L)
            post_gpu = cache[hook][:, -1, :]
            W_out_gpu= model.blocks[L].mlp.W_out

            post = post_gpu.float().cpu()
            W_out= W_out_gpu.float().cpu()

            for n_idx in n_list:
                vec = W_out[n_idx]
                a_n = post[:, n_idx]
                logits_n = (a_n[:, None] * vec[None, :]) @ model.W_U.float().cpu()
                kl = kl_div(gt_last, logits_n).tolist()
                kl_buf[f"L{L}-N{n_idx}"].extend(kl)

        del cache; torch.cuda.empty_cache()

    # ------------------- save JSON -------------------
    out_dir = pathlib.Path(cfg["out_dir"]); out_dir.mkdir(exist_ok=True)
    stamp   = time.strftime("%Y%m%d_%H%M%S")
    save_json({"cfg": cfg, "kl": kl_buf},
              out_dir, f"neuron_probe_{stamp}")

    # ------------------- collect baselines -------------------
    box, lbl = [], []
    try:
        rq1_json = sorted(out_dir.glob("rq1_*json"))[-1]
        rec      = json.load(open(rq1_json))["rec"]
        box.append(np.array(rec["kl_logit"])); lbl.append("Logit‑Lens")
        box.append(np.array(rec["kl_ov"]));    lbl.append("Head‑OV")
    except Exception:
        print("⚠️  RQ1 JSON not found — baselines skipped.")

    for k in key_list:
        box.append(kl_buf[k]); lbl.append(k)

    # ------------------- plot -------------------
    plt.figure(figsize=(6.4,3.6))
    parts = plt.violinplot(box, showmeans=False, showextrema=False)

    colors = ["#CCCCCC", "#8FAADC"] + ["#E6A23C"]*len(key_list)
    for pcol, pc in zip(parts['bodies'], colors):
        pcol.set_facecolor(pc); pcol.set_alpha(0.7)

    for i, data in enumerate(box, start=1):
        plt.scatter(i, np.mean(data), marker="D", color="black", s=18, zorder=3)
        plt.scatter(i, np.median(data), marker="_", color="red", s=60, zorder=3)

    plt.xticks(range(1, len(lbl)+1), lbl, rotation=30, ha='right')
    plt.ylabel("KL divergence (nats) ↓")
    plt.title("Single‑Neuron OV‑Lens vs Baselines\n(↓ better than Logit‑Lens)")
    plt.axhline(np.mean(box[0]), ls="--", color="gray", lw=1, label="Logit mean")
    plt.legend(loc="upper right", frameon=False, fontsize=9)
    plt.tight_layout()
    png = out_dir / "neuron_probe_violin.png"
    plt.savefig(png, dpi=300); plt.close()
    print("🎻 saved", png.name)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    probe(cfg)
