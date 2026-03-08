"""
Experiment 4: Linear probing analysis.
Train linear probes to predict numeric values from hidden states,
comparing accuracy across number forms and vigesimal vs decimal ranges.
Also: analyze the structure of the embedding space in detail.
"""

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
from scipy.spatial.distance import cosine, pdist, squareform
import random

random.seed(42)
np.random.seed(42)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "datasets" / "french_numbers"
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def load_representations():
    """Reconstruct representations by re-running extraction (or load cached)."""
    cache_path = RESULTS_DIR / "hidden_states_cache.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["representations"].item(), data["layer_indices"].tolist()

    # Need to re-extract — run model
    from transformers import AutoTokenizer, AutoModelForCausalLM

    MODEL_NAME = "mistralai/Mistral-7B-v0.3"
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()

    with open(DATA_DIR / "french_numbers_0_999.json") as f:
        all_numbers = json.load(f)
    numbers_0_99 = [n for n in all_numbers if n["number"] <= 99]

    with open(DATA_DIR / "vigesimal_subset_70_99.json") as f:
        vigesimal = json.load(f)
    belgian_lookup = {v["number"]: v["belgian_french_word"] for v in vigesimal}
    for entry in numbers_0_99:
        if entry["number"] in belgian_lookup:
            entry["belgian_french_word"] = belgian_lookup[entry["number"]]

    n_layers = model.config.num_hidden_layers
    layer_indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]

    # Extract all hidden states
    representations = {}
    for entry in numbers_0_99:
        num = entry["number"]
        forms = {"french": entry["french_word"], "digit": str(num)}
        if "belgian_french_word" in entry:
            forms["belgian"] = entry["belgian_french_word"]

        for form_name, form_text in forms.items():
            prompt = f"Le nombre est {form_text}."
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            for li in layer_indices:
                h = outputs.hidden_states[li][0, -1, :].cpu().float().numpy()
                key = (num, form_name, li)
                representations[key] = h

    # Cache
    np.savez(cache_path, representations=representations, layer_indices=np.array(layer_indices))
    del model
    torch.cuda.empty_cache()
    return representations, layer_indices


def linear_probe_analysis(representations, layer_indices):
    """Train linear probes to predict number values from hidden states."""
    print("\n" + "=" * 60)
    print("LINEAR PROBE ANALYSIS")
    print("=" * 60)

    results = {}
    last_layer = max(layer_indices)

    for layer_idx in layer_indices:
        layer_label = f"layer_{layer_idx}"
        results[layer_label] = {}

        for form in ["french", "digit", "belgian"]:
            # Collect data for this form and layer
            X = []
            y = []
            for (num, f, li), h in representations.items():
                if f == form and li == layer_idx:
                    X.append(h)
                    y.append(num)
            if not X:
                continue

            X = np.array(X)
            y = np.array(y)

            # 5-fold cross-validated Ridge regression
            ridge = Ridge(alpha=1.0)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            maes = []
            r2s = []
            vig_maes = []
            dec_maes = []

            for train_idx, test_idx in kf.split(X):
                ridge.fit(X[train_idx], y[train_idx])
                preds = ridge.predict(X[test_idx])
                mae = mean_absolute_error(y[test_idx], preds)
                r2 = r2_score(y[test_idx], preds)
                maes.append(mae)
                r2s.append(r2)

                # Split by vigesimal
                for i in test_idx:
                    err = abs(preds[list(test_idx).index(i)] - y[i])
                    if 70 <= y[i] <= 99:
                        vig_maes.append(err)
                    elif 20 <= y[i] <= 69:
                        dec_maes.append(err)

            results[layer_label][form] = {
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes)),
                "r2_mean": float(np.mean(r2s)),
                "r2_std": float(np.std(r2s)),
                "vigesimal_mae": float(np.mean(vig_maes)) if vig_maes else None,
                "decimal_mae": float(np.mean(dec_maes)) if dec_maes else None,
            }

            if layer_idx == last_layer:
                print(f"\n  Layer {layer_idx}, {form}:")
                print(f"    MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f}")
                print(f"    R²: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
                if vig_maes:
                    print(f"    Vigesimal MAE: {np.mean(vig_maes):.2f}")
                if dec_maes:
                    print(f"    Decimal MAE: {np.mean(dec_maes):.2f}")

    return results


def representational_similarity_analysis(representations, layer_indices):
    """RSA: Compare representational geometry across forms."""
    print("\n" + "=" * 60)
    print("REPRESENTATIONAL SIMILARITY ANALYSIS (RSA)")
    print("=" * 60)

    last_layer = max(layer_indices)
    rsa_results = {}

    # Get representations for each form at last layer
    forms_data = {}
    for (num, form, li), h in representations.items():
        if li == last_layer:
            if form not in forms_data:
                forms_data[form] = {"nums": [], "hiddens": []}
            forms_data[form]["nums"].append(num)
            forms_data[form]["hiddens"].append(h)

    # Sort by number
    for form in forms_data:
        order = np.argsort(forms_data[form]["nums"])
        forms_data[form]["nums"] = [forms_data[form]["nums"][i] for i in order]
        forms_data[form]["hiddens"] = [forms_data[form]["hiddens"][i] for i in order]

    # Compute RDMs (Representational Dissimilarity Matrices) for each form
    rdms = {}
    for form, data in forms_data.items():
        H = np.array(data["hiddens"])
        rdm = squareform(pdist(H, metric="cosine"))
        rdms[form] = rdm

    # Also compute ideal numeric RDM
    nums = np.array(forms_data["french"]["nums"])  # All forms have same numbers
    ideal_rdm = squareform(pdist(nums.reshape(-1, 1), metric="euclidean"))
    # Normalize
    ideal_rdm = ideal_rdm / ideal_rdm.max()

    # RSA: correlate RDMs
    print("\n  Pairwise RSA correlations (Spearman):")
    all_forms = list(rdms.keys()) + ["ideal_numeric"]
    rdms["ideal_numeric"] = ideal_rdm

    rsa_matrix = {}
    for f1 in all_forms:
        for f2 in all_forms:
            if f1 >= f2:
                continue
            v1 = squareform(rdms[f1])
            v2 = squareform(rdms[f2])
            rho, p = stats.spearmanr(v1, v2)
            rsa_matrix[f"{f1}_vs_{f2}"] = {"rho": float(rho), "p": float(p)}
            print(f"    {f1} vs {f2}: ρ = {rho:.3f} (p = {p:.2e})")

    # Visualize RDMs
    form_labels = ["french", "digit", "belgian", "ideal_numeric"] if "belgian" in rdms else ["french", "digit", "ideal_numeric"]
    fig, axes = plt.subplots(1, len(form_labels), figsize=(5 * len(form_labels), 4))
    for i, form in enumerate(form_labels):
        ax = axes[i]
        im = ax.imshow(rdms[form], cmap="viridis", aspect="auto")
        ax.set_title(f"RDM: {form}")
        ax.set_xlabel("Number index")
        ax.set_ylabel("Number index")
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rdm_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/rdm_comparison.png")

    return rsa_matrix


def vigesimal_structure_analysis(representations, layer_indices):
    """Analyze how the vigesimal structure affects representation geometry."""
    print("\n" + "=" * 60)
    print("VIGESIMAL STRUCTURE ANALYSIS")
    print("=" * 60)

    last_layer = max(layer_indices)

    # Get French representations
    french_data = {}
    for (num, form, li), h in representations.items():
        if form == "french" and li == last_layer:
            french_data[num] = h

    # For numbers in vigesimal range, check if representations cluster
    # by their "base component" (60+x for 70s, 80+x for 80s/90s)
    # vs by their actual numeric value

    # Compute: for vigesimal numbers, is the nearest neighbor in embedding space
    # numerically close, or does it share the same vigesimal base?
    print("\n  Nearest neighbor analysis for vigesimal numbers:")

    vig_analysis = []
    for num in range(70, 100):
        if num not in french_data:
            continue
        h = french_data[num]

        # Find nearest neighbor in full set
        min_dist = float("inf")
        nn = num
        for other_num, other_h in french_data.items():
            if other_num == num:
                continue
            d = cosine(h, other_h)
            if d < min_dist:
                min_dist = d
                nn = other_num

        # Determine if NN shares vigesimal base
        same_decade = (num // 10 == nn // 10)
        shares_unit = (num % 10 == nn % 10)
        numeric_dist = abs(num - nn)

        # For 70s: linguistic base is 60+x, so check if NN is in 60s
        if 70 <= num <= 79:
            shares_ling_base = (60 <= nn <= 69)  # same "soixante-" prefix
        elif 80 <= num <= 99:
            shares_ling_base = (80 <= nn <= 99)  # same "quatre-vingt-" prefix
        else:
            shares_ling_base = same_decade

        vig_analysis.append({
            "number": num,
            "nearest_neighbor": nn,
            "cosine_distance": float(min_dist),
            "numeric_distance": numeric_dist,
            "same_decade": same_decade,
            "shares_unit_digit": shares_unit,
            "shares_linguistic_base": shares_ling_base,
        })

        if numeric_dist > 3:  # Only print interesting cases
            print(f"    {num} → NN is {nn} (dist={numeric_dist}, cos={min_dist:.4f}, "
                  f"ling_base={'yes' if shares_ling_base else 'no'})")

    # Summary stats
    ling_base_count = sum(1 for v in vig_analysis if v["shares_linguistic_base"])
    numeric_close = sum(1 for v in vig_analysis if v["numeric_distance"] <= 2)
    print(f"\n  Summary (vigesimal range 70-99):")
    print(f"    NN shares linguistic base: {ling_base_count}/{len(vig_analysis)} ({ling_base_count/len(vig_analysis):.1%})")
    print(f"    NN within ±2 numerically: {numeric_close}/{len(vig_analysis)} ({numeric_close/len(vig_analysis):.1%})")

    # Compare with decimal range control
    dec_analysis = []
    for num in range(20, 70):
        if num not in french_data:
            continue
        h = french_data[num]
        min_dist = float("inf")
        nn = num
        for other_num, other_h in french_data.items():
            if other_num == num:
                continue
            d = cosine(h, other_h)
            if d < min_dist:
                min_dist = d
                nn = other_num
        dec_analysis.append({
            "number": num,
            "nearest_neighbor": nn,
            "numeric_distance": abs(num - nn),
        })

    dec_close = sum(1 for d in dec_analysis if d["numeric_distance"] <= 2)
    print(f"\n  Control (decimal range 20-69):")
    print(f"    NN within ±2 numerically: {dec_close}/{len(dec_analysis)} ({dec_close/len(dec_analysis):.1%})")

    # Statistical comparison
    vig_nn_dists = [v["numeric_distance"] for v in vig_analysis]
    dec_nn_dists = [d["numeric_distance"] for d in dec_analysis]
    u_stat, p_val = stats.mannwhitneyu(vig_nn_dists, dec_nn_dists, alternative="greater")
    print(f"\n  Mann-Whitney U test (vigesimal NN distance > decimal):")
    print(f"    U = {u_stat:.1f}, p = {p_val:.4f}")
    print(f"    Vigesimal mean NN dist: {np.mean(vig_nn_dists):.2f}")
    print(f"    Decimal mean NN dist: {np.mean(dec_nn_dists):.2f}")

    # Plot: NN distance by number
    fig, ax = plt.subplots(figsize=(14, 5))
    all_nums = list(range(100))
    all_nn_dists = []
    for num in all_nums:
        if num in french_data:
            h = french_data[num]
            min_dist = float("inf")
            nn = num
            for other_num, other_h in french_data.items():
                if other_num == num:
                    continue
                d = cosine(h, other_h)
                if d < min_dist:
                    min_dist = d
                    nn = other_num
            all_nn_dists.append(abs(num - nn))
        else:
            all_nn_dists.append(0)

    colors = ["indianred" if 70 <= n <= 99 else "steelblue" for n in all_nums]
    ax.bar(all_nums, all_nn_dists, color=colors, alpha=0.7)
    ax.axvspan(70, 99, alpha=0.1, color="red")
    ax.set_xlabel("Number")
    ax.set_ylabel("Nearest Neighbor Numeric Distance")
    ax.set_title("Embedding Space NN Distance (red = vigesimal range)")
    ax.grid(True, alpha=0.3, axis="y")

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="steelblue", alpha=0.7, label="Decimal (0-69)"),
        mpatches.Patch(color="indianred", alpha=0.7, label="Vigesimal (70-99)")
    ])
    plt.savefig(PLOTS_DIR / "nn_distance_by_number.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/nn_distance_by_number.png")

    return {
        "vigesimal_nn_analysis": vig_analysis,
        "vigesimal_mean_nn_dist": float(np.mean(vig_nn_dists)),
        "decimal_mean_nn_dist": float(np.mean(dec_nn_dists)),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(p_val),
    }


def layer_progression_analysis(representations, layer_indices):
    """Analyze how number representations develop across layers."""
    print("\n" + "=" * 60)
    print("LAYER PROGRESSION ANALYSIS")
    print("=" * 60)

    layer_metrics = []
    for li in layer_indices:
        # Get French representations at this layer
        nums = []
        hiddens = []
        for (num, form, layer), h in representations.items():
            if form == "french" and layer == li:
                nums.append(num)
                hiddens.append(h)

        if not hiddens:
            continue

        X = np.array(hiddens)
        y = np.array(nums)

        # Fit probe
        ridge = Ridge(alpha=1.0)
        scores = cross_val_score(ridge, X, y, cv=5, scoring="neg_mean_absolute_error")
        mae = -scores.mean()

        # Spearman correlation of pairwise distances
        from scipy.spatial.distance import pdist
        cos_dists = pdist(X, metric="cosine")
        num_dists = pdist(y.reshape(-1, 1), metric="euclidean")
        rho, _ = stats.spearmanr(cos_dists, num_dists)

        layer_metrics.append({
            "layer": li,
            "probe_mae": float(mae),
            "spearman_rho": float(rho),
        })
        print(f"  Layer {li}: MAE={mae:.2f}, Spearman ρ={rho:.3f}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    layers = [m["layer"] for m in layer_metrics]
    maes = [m["probe_mae"] for m in layer_metrics]
    rhos = [m["spearman_rho"] for m in layer_metrics]

    color1 = "steelblue"
    color2 = "indianred"
    ax1.plot(layers, maes, "o-", color=color1, label="Probe MAE", linewidth=2)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Probe MAE", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(layers, rhos, "s-", color=color2, label="Spearman ρ", linewidth=2)
    ax2.set_ylabel("Spearman ρ (distance correlation)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title("Number Representation Quality Across Layers")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    ax1.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / "layer_progression.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/layer_progression.png")

    return layer_metrics


def main():
    representations, layer_indices = load_representations()

    # Linear probe analysis
    probe_results = linear_probe_analysis(representations, layer_indices)

    # RSA
    rsa_results = representational_similarity_analysis(representations, layer_indices)

    # Vigesimal structure analysis
    vig_results = vigesimal_structure_analysis(representations, layer_indices)

    # Layer progression
    layer_results = layer_progression_analysis(representations, layer_indices)

    # Save all
    all_results = {
        "linear_probes": probe_results,
        "rsa": rsa_results,
        "vigesimal_structure": {
            "vigesimal_mean_nn_dist": vig_results["vigesimal_mean_nn_dist"],
            "decimal_mean_nn_dist": vig_results["decimal_mean_nn_dist"],
            "mann_whitney_p": vig_results["mann_whitney_p"],
        },
        "layer_progression": layer_results,
    }
    with open(RESULTS_DIR / "probing_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nAll probing results saved to {RESULTS_DIR}/probing_results.json")


if __name__ == "__main__":
    main()
