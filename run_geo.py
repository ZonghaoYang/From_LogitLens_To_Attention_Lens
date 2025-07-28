#!/usr/bin/env python
# ─── src/run_rq1_3_geo.py ────────────────────────────────────────────────

from __future__ import annotations
import argparse, yaml, pathlib, json, random
from typing import List, Dict, Set

import numpy as np, torch, matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate
from transformer_lens import utils as tl_utils
from thesis_utils     import set_seed, load_model, get_prompts
from collections import OrderedDict
import pandas as pd

def make_prompt_tables(prompts: List[str],
                       token_strings: List[str],
                       base_lp: np.ndarray,
                       plus_lp: np.ndarray,
                       minus_lp: np.ndarray,
                       layer: int,
                       neuron_id: int,
                       out_dir: pathlib.Path,
                       top_k: int = 5):

    rows_markdown = []
    csv_records   = []
    for i, p in enumerate(prompts):
        delta = plus_lp[i] - base_lp[i]
        idx   = np.argsort(-delta)[:top_k]
        for j in idx:
            rows_markdown.append([
                i+1, p, token_strings[j],
                f"{delta[j]:+.3f}",
                f"{(minus_lp[i]-base_lp[i])[j]:+.3f}"
            ])
            csv_records.append( OrderedDict(
                prompt_id = i+1,
                prompt    = p,
                token     = token_strings[j],
                delta_pos = float(delta[j]),
                delta_neg = float((minus_lp[i]-base_lp[i])[j])
            ))

    md_hdr = ["| # | Prompt (trunc) | Token | Δ log‑p (+) | Δ log‑p (‑) |",
              "|:-:|:--|:--|--:|--:|"]
    md_body= ["|{}|{}|{}|{}|{}|".format(*r) for r in rows_markdown]
    (out_dir / f"prompt_diff_L{layer}.md").write_text("\n".join(md_hdr+md_body), encoding="utf-8")

    pd.DataFrame(csv_records).to_csv(out_dir / f"prompt_diff_L{layer}.csv", index=False)


EU_COUNTRIES = [
    "Albania","Andorra","Armenia","Austria","Azerbaijan","Belarus","Belgium",
    "Bosnia","Bulgaria","Croatia","Cyprus","Czech","Denmark","Estonia",
    "Finland","France","Georgia","Germany","Greece","Hungary","Iceland","Ireland",
    "Italy","Kazakhstan","Kosovo","Latvia","Liechtenstein","Lithuania","Luxembourg",
    "Malta","Moldova","Monaco","Montenegro","Netherlands","North Macedonia",
    "Norway","Poland","Portugal","Romania","Russia","San Marino","Serbia",
    "Slovakia","Slovenia","Spain","Sweden","Switzerland","Turkey","Ukraine",
    "United Kingdom","Vatican"
]
EU_CAPITALS = [
    "Vienna","Brussels","Sofia","Zagreb","Prague","Copenhagen","Tallinn",
    "Helsinki","Paris","Berlin","Athens","Budapest","Reykjavik","Dublin",
    "Rome","Riga","Vilnius","Valletta","Chisinau","Monaco","Podgorica",
    "Amsterdam","Oslo","Warsaw","Lisbon","Bucharest","Belgrade","Bratislava",
    "Ljubljana","Madrid","Stockholm","Bern","Kyiv","Ankara","London","Moscow",
    "Tirana","Pristina","Sarajevo","Yerevan","Baku","Tbilisi"
]

def neuron_logit_vec(model, layer:int, neuron:int) -> torch.Tensor:
    W_out = model.blocks[layer].mlp.W_out
    W     = W_out.weight if isinstance(W_out, torch.nn.Linear) else W_out
    vec_d = W.t()[neuron] if W.shape[0] == model.cfg.d_model else W[neuron]
    return (vec_d @ model.W_U).to(torch.float32).detach()

def load_geo_vocab(model) -> Set[int]:
    tok = model.tokenizer; vocab=set()
    for w in EU_COUNTRIES+EU_CAPITALS:
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if len(ids)==1: vocab.add(ids[0])
    return vocab

def find_geo_neurons(model, layer:int, geo_ids:Set[int],
                     top_n:int=10, kurt_th:float=5.0) -> List[Dict]:
    cand=[]
    for n in range(model.cfg.d_mlp):
        vec = neuron_logit_vec(model, layer, n).cpu().numpy()
        if stats.kurtosis(vec) < kurt_th:
            continue
        score = vec[list(geo_ids)].clip(min=0).sum()
        if score > 0:
            top_geo = [model.to_string(i)
                       for i in np.argsort(vec)[::-1] if i in geo_ids][:5]
            cand.append({"neuron": n, "score": float(score), "top_geo": top_geo})
    return sorted(cand, key=lambda x: -x["score"])[:top_n]

def patch_neuron_once(model, ids, layer, neuron_id, delta_sigma=+5.0):
    hook_name = tl_utils.get_act_name("mlp_post", layer)
    with torch.no_grad():
        _, cache = model.run_with_cache(ids[:2,:],
                                        names_filter=lambda n: n==hook_name)
    sigma = cache[hook_name][:,:,neuron_id].std().item() + 1e-5
    delta = delta_sigma * sigma
    def add_delta(act, hook):
        act[:,:,neuron_id] += delta
        return act
    with torch.no_grad():
        return model.run_with_hooks(ids, fwd_hooks=[(hook_name, add_delta)])


def token_bar(tokens, boosts, png_path):
    plt.figure(figsize=(6,3.5))
    plt.barh(range(len(tokens))[::-1], boosts)
    plt.yticks(range(len(tokens))[::-1], tokens, fontsize=9)
    plt.xlabel("Relative prob ↑  (+5σ)")
    plt.title("Geo‑Neuron token boost")
    plt.tight_layout(); plt.savefig(png_path, dpi=300); plt.close()

def save_tables(tokens, base_p, plus_p, minus_p,
                layer:int, cfg:dict, out_dir:pathlib.Path):
    import pandas as pd
    df = pd.DataFrame({
        "token": tokens,
        "P(base)":  base_p,
        "P(+5σ)":   plus_p,
        "P(-5σ)":   minus_p,
        "Δ+ (%)":   100 * (plus_p / base_p - 1),
        "Δ- (%)":   100 * (minus_p / base_p - 1),
    }).round(4)

    csv_path = out_dir / f"geo_neuron_L{layer}.csv"
    df.to_csv(csv_path, index=False)

    if cfg.get("save_markdown", False):
        md_path = out_dir / f"geo_neuron_L{layer}.md"
        md_path.write_text(df.to_markdown(index=False), encoding="utf-8")

    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))

def render_markdown_demo(prompts, tok_str, base, plus, minus,
                         top_n:int, thresh:float) -> str:
    rows=[]
    for i,p in enumerate(prompts):
        delta = plus[i] - base[i]
        sel   = np.argsort(np.abs(delta))[::-1][:top_n]
        for j in sel:
            if abs(delta[j]) < thresh: continue
            rows.append([i+1,
                         p.replace("|","\\|")[:60]+"…",
                         tok_str[j],
                         f"{delta[j]:+.3f}",
                         f"{(minus[i]-base[i])[j]:+.3f}"])
    if not rows:
        return "_No token exceeded threshold._\n"
    hdr = ["| # | Prompt (trunc) | Token | Δ log‑p (+) | Δ log‑p (‑) |",
           "|:-:|:--|:--|--:|--:|"]
    body = ["|{}|{}|{}|{}|{}|".format(*r) for r in rows]
    return "\n".join(hdr+body)+"\n"

def render_ansi_diff(prompts, tok_str, base, plus, top_n:int, thresh:float) -> str:
    lines=[]
    for i,p in enumerate(prompts):
        delta = plus[i] - base[i]
        sel   = np.argsort(np.abs(delta))[::-1][:top_n]
        coloured=[]
        for j in sel:
            if abs(delta[j]) < thresh: continue
            colour = "\033[1;32m" if delta[j]>0 else "\033[1;31m"
            coloured.append(f"{colour}{tok_str[j]} ({delta[j]:+.2f})\033[0m")
        if coloured:
            lines.append(f"[{i+1}] {p}\n    "+", ".join(coloured))
    return "\n".join(lines) if lines else "No token exceeded threshold."

def main(cfg):
    set_seed(cfg["seed"])
    model = load_model(cfg["model"], dtype=cfg.get("dtype", "bfloat16"))
    all_layers = cfg.get("layers", [cfg.get("layer")])
    out_dir    = pathlib.Path(cfg["out_dir"]); out_dir.mkdir(exist_ok=True)

    prompts = get_prompts(dataset   = cfg["dataset"],
                          n_prompts = cfg["n_prompts"],
                          max_len   = cfg["max_len"],
                          seed      = cfg["seed"])
    geo_words = {w.lower() for w in EU_COUNTRIES+EU_CAPITALS}
    demo = [p for p in prompts if any(g in p.lower() for g in geo_words)]
    demo = demo[:cfg["n_demo"]] + prompts[:max(0, cfg["n_demo"]-len(demo))]
    tok   = model.tokenizer
    ids   = tok(demo, padding=True, truncation=True,
                max_length=cfg["max_len"],
                return_tensors="pt").input_ids.to(model.W_U.device)

    geo_ids  = sorted(list(load_geo_vocab(model)))
    tok_str  = [model.to_string(i) for i in geo_ids]
    tgt_ids  = torch.tensor(geo_ids, device=model.W_U.device)

    for layer in all_layers:
        # ——— 1. Find geo-related neurons ———
        geo_neurons = find_geo_neurons(model, layer, set(geo_ids),
                                      top_n=5, kurt_th=cfg["kurtosis_th"])
        if not geo_neurons:
            print(f"[Layer {layer}] ‼ No geo-neuron passed the threshold. Skipping.")
            continue
        neuron = geo_neurons[0]
        print(f"\n★  Geo-Neuron  L{layer}-N{neuron['neuron']}  "
              f"score={neuron['score']:.1f}")
        print("   Top geo tokens:", ", ".join(neuron["top_geo"]))



        with torch.no_grad():
            logits_base  = model(ids)[:,-1,:]
        logits_plus  = patch_neuron_once(model, ids, layer,
                                         neuron["neuron"], +cfg["delta_sigma"])[:,-1,:]
        logits_minus = patch_neuron_once(model, ids, layer,
                                         neuron["neuron"], -cfg["delta_sigma"])[:,-1,:]

        base_lp  = torch.log_softmax(logits_base.float(),  -1)[:, tgt_ids].cpu().numpy()
        plus_lp  = torch.log_softmax(logits_plus.float(),  -1)[:, tgt_ids].cpu().numpy()
        minus_lp = torch.log_softmax(logits_minus.float(), -1)[:, tgt_ids].cpu().numpy()

        mean_base  = base_lp.mean(0); mean_plus = plus_lp.mean(0); mean_minus = minus_lp.mean(0)
        boosts_rel = np.exp(mean_plus - mean_base) - 1

        top_idx   = np.argsort(boosts_rel)[-15:][::-1]
        token_bar([tok_str[i] for i in top_idx],
                  boosts_rel[top_idx],
                  out_dir / f"geo_neuron_L{layer}.png")

        save_tables([tok_str[i] for i in top_idx],
                    np.exp(mean_base[top_idx]),
                    np.exp(mean_plus[top_idx]),
                    np.exp(mean_minus[top_idx]),
                    layer, cfg, out_dir)

        json.dump({
            "cfg": cfg, "layer": layer, "neuron": neuron,
            "prob_base":  mean_base.tolist(),
            "prob_plus":  mean_plus.tolist(),
            "prob_minus": mean_minus.tolist(),
            "ratio_plus": (np.exp(mean_plus - mean_base)).tolist(),
            "demo_prompts": demo
        }, open(out_dir / f"rq1_3_geo_L{layer}.json", "w"), indent=2)

        if cfg.get("save_markdown", True):
            md = render_markdown_demo(demo, tok_str,
                                      base_lp, plus_lp, minus_lp,
                                      top_n   = cfg.get("report_top_n", 5),
                                      thresh  = cfg.get("prob_thresh", 0.05))
            (out_dir / f"rq1_3_geo_L{layer}.md").write_text(md, encoding="utf-8")

        if cfg.get("save_ansi", True):
            ansi = render_ansi_diff(demo, tok_str,
                                    base_lp, plus_lp,
                                    top_n   = cfg.get("report_top_n", 5),
                                    thresh  = cfg.get("prob_thresh", 0.05))
            (out_dir / f"rq1_3_geo_L{layer}.ansi").write_text(ansi, encoding="utf-8")


        make_prompt_tables(
            prompts      = demo,
            token_strings= tok_str,
            base_lp      = base_lp,
            plus_lp      = plus_lp,
            minus_lp     = minus_lp,
            layer        = layer,
            neuron_id    = neuron["neuron"],
            out_dir      = out_dir,
            top_k        = cfg.get("report_top_n", 5)
        )


        print(f"✅  Results for Layer {layer} have been saved to {out_dir}")


# ───────────────────────────  CLI  ────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    main(cfg)
