"""
Experiment 2: Representational analysis of French number words in LLMs.
Uses a local transformer model to extract hidden state representations
and analyze how French numbers are encoded internally.

Key questions:
- Do vigesimal French numbers (70-99) cluster differently from decimal (20-69)?
- Does Belgian French produce cleaner representations?
- How do French number representations compare to digit representations?
"""

import json
import os
import sys
import random
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = Path(__file__).parent.parent / "datasets" / "french_numbers"
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Use Mistral 7B — strong French language model, developed in France
MODEL_NAME = "mistralai/Mistral-7B-v0.3"


def load_model():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")
    return tokenizer, model


def get_last_token_hidden_states(text, tokenizer, model, layer_indices=None):
    """Extract hidden states at the last token position for specified layers."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # tuple of (n_layers+1, batch, seq, hidden)
    last_token_states = {}
    if layer_indices is None:
        # Sample layers: early, middle, late
        n_layers = len(hidden_states) - 1
        layer_indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]

    for li in layer_indices:
        if li < len(hidden_states):
            # Last token hidden state
            h = hidden_states[li][0, -1, :].cpu().float().numpy()
            last_token_states[li] = h
    return last_token_states


def extract_representations(tokenizer, model, numbers_data):
    """Extract representations for all number forms."""
    n_layers = model.config.num_hidden_layers
    # Sample 5 layers across the model
    layer_indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]

    results = []
    for i, entry in enumerate(numbers_data):
        if i % 20 == 0:
            print(f"  Processing number {i}/{len(numbers_data)}...")

        num = entry["number"]
        forms = {
            "french": entry["french_word"],
            "digit": str(num),
        }
        if "belgian_french_word" in entry:
            forms["belgian"] = entry["belgian_french_word"]

        # Create contextual prompts for each form
        for form_name, form_text in forms.items():
            # Use a simple template to contextualize the number
            prompt = f"Le nombre est {form_text}."
            states = get_last_token_hidden_states(prompt, tokenizer, model, layer_indices)
            for layer_idx, hidden in states.items():
                results.append({
                    "number": num,
                    "form": form_name,
                    "text": form_text,
                    "layer": layer_idx,
                    "uses_vigesimal": entry.get("uses_vigesimal", False),
                    "category": entry.get("category", ""),
                    "hidden_state": hidden,
                })
    return results, layer_indices


def analyze_representations(results, layer_indices):
    """Analyze and visualize number representations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from scipy.stats import spearmanr
    from scipy.spatial.distance import cosine

    analysis = {}

    # Focus on the last layer for main analysis
    last_layer = max(layer_indices)

    for layer_idx in [last_layer, layer_indices[len(layer_indices) // 2]]:
        layer_label = "last" if layer_idx == last_layer else "middle"
        layer_results = [r for r in results if r["layer"] == layer_idx]

        # Group by form
        forms = {}
        for r in layer_results:
            form = r["form"]
            if form not in forms:
                forms[form] = {"numbers": [], "hiddens": [], "vigesimal": []}
            forms[form]["numbers"].append(r["number"])
            forms[form]["hiddens"].append(r["hidden_state"])
            forms[form]["vigesimal"].append(r["uses_vigesimal"])

        # --- Analysis 1: Ordinal correlation ---
        # For each form, compute Spearman correlation between
        # pairwise cosine distances and numeric distances
        print(f"\n  [Layer {layer_idx} ({layer_label})] Ordinal correlation analysis:")
        ordinal_results = {}
        for form_name, data in forms.items():
            hiddens = np.array(data["hiddens"])
            numbers = np.array(data["numbers"])
            n = len(numbers)
            if n < 3:
                continue

            # Compute pairwise distances
            cos_dists = []
            num_dists = []
            for i in range(n):
                for j in range(i + 1, n):
                    cos_dists.append(cosine(hiddens[i], hiddens[j]))
                    num_dists.append(abs(numbers[i] - numbers[j]))

            rho, pval = spearmanr(num_dists, cos_dists)
            ordinal_results[form_name] = {"spearman_rho": rho, "p_value": pval}
            print(f"    {form_name}: Spearman ρ = {rho:.3f} (p = {pval:.2e})")

        # --- Analysis 2: Vigesimal vs Decimal representation quality ---
        print(f"\n  [Layer {layer_idx} ({layer_label})] Vigesimal vs Decimal analysis:")
        for form_name, data in forms.items():
            hiddens = np.array(data["hiddens"])
            numbers = np.array(data["numbers"])
            vig_mask = np.array(data["vigesimal"])

            if not any(vig_mask):
                continue

            # For each number, compute distance to nearest neighbors in embedding space
            # vs. actual numeric neighbors
            vig_hiddens = hiddens[vig_mask]
            dec_hiddens = hiddens[~vig_mask]
            vig_numbers = numbers[vig_mask]
            dec_numbers = numbers[~vig_mask]

            # Compute "representation error": for each number, find the number
            # whose representation is closest and measure how far it is numerically
            def nearest_neighbor_error(h_set, n_set):
                errors = []
                for i in range(len(n_set)):
                    min_dist = float("inf")
                    nn_num = n_set[i]
                    for j in range(len(n_set)):
                        if i == j:
                            continue
                        d = cosine(h_set[i], h_set[j])
                        if d < min_dist:
                            min_dist = d
                            nn_num = n_set[j]
                    errors.append(abs(n_set[i] - nn_num))
                return np.mean(errors), np.std(errors)

            if len(vig_numbers) > 2:
                vig_err, vig_std = nearest_neighbor_error(vig_hiddens, vig_numbers)
                print(f"    {form_name} vigesimal NN error: {vig_err:.1f} ± {vig_std:.1f}")
            if len(dec_numbers) > 2:
                dec_err, dec_std = nearest_neighbor_error(dec_hiddens, dec_numbers)
                print(f"    {form_name} decimal NN error: {dec_err:.1f} ± {dec_std:.1f}")

        # --- Visualization: t-SNE ---
        print(f"\n  [Layer {layer_idx} ({layer_label})] Creating t-SNE visualization...")

        # Collect all representations for this layer
        all_hiddens = []
        all_labels = []
        all_forms_list = []
        all_vig = []
        for r in layer_results:
            all_hiddens.append(r["hidden_state"])
            all_labels.append(r["number"])
            all_forms_list.append(r["form"])
            all_vig.append(r["uses_vigesimal"])

        all_hiddens = np.array(all_hiddens)

        # PCA first then t-SNE for stability
        pca = PCA(n_components=min(50, all_hiddens.shape[0] - 1, all_hiddens.shape[1]))
        pca_result = pca.fit_transform(all_hiddens)

        tsne = TSNE(n_components=2, perplexity=min(30, len(all_hiddens) // 4),
                     random_state=42, max_iter=1000)
        tsne_result = tsne.fit_transform(pca_result)

        # Plot by form type
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        form_colors = {"french": "blue", "digit": "green", "belgian": "red"}
        form_names = list(set(all_forms_list))

        # Plot 1: Colored by form
        ax = axes[0]
        for form_name in form_names:
            mask = [f == form_name for f in all_forms_list]
            idx = [i for i, m in enumerate(mask) if m]
            ax.scatter(tsne_result[idx, 0], tsne_result[idx, 1],
                      c=form_colors.get(form_name, "gray"), label=form_name,
                      alpha=0.6, s=30)
        ax.set_title(f"Layer {layer_idx}: Colored by Number Form")
        ax.legend()
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        # Plot 2: Colored by numeric value
        ax = axes[1]
        sc = ax.scatter(tsne_result[:, 0], tsne_result[:, 1],
                       c=all_labels, cmap="viridis", alpha=0.6, s=30)
        plt.colorbar(sc, ax=ax, label="Number Value")
        ax.set_title(f"Layer {layer_idx}: Colored by Numeric Value")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        # Plot 3: Colored by vigesimal status (French only)
        ax = axes[2]
        french_mask = [f == "french" for f in all_forms_list]
        french_idx = [i for i, m in enumerate(french_mask) if m]
        french_vig = [all_vig[i] for i in french_idx]
        colors = ["red" if v else "blue" for v in french_vig]
        ax.scatter(tsne_result[french_idx, 0], tsne_result[french_idx, 1],
                  c=colors, alpha=0.6, s=30)
        # Add number labels
        for i, idx in enumerate(french_idx):
            ax.annotate(str(all_labels[idx]), (tsne_result[idx, 0], tsne_result[idx, 1]),
                       fontsize=6, alpha=0.7)
        ax.set_title(f"Layer {layer_idx}: French Only (red=vigesimal, blue=decimal)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"tsne_layer_{layer_label}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {PLOTS_DIR}/tsne_layer_{layer_label}.png")

        # --- PCA variance analysis ---
        pca_full = PCA(n_components=min(10, all_hiddens.shape[0] - 1))
        pca_full.fit(all_hiddens)
        explained = pca_full.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(explained)), explained)
        ax.set_xlabel("PCA Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title(f"Layer {layer_idx}: PCA Explained Variance")
        plt.savefig(PLOTS_DIR / f"pca_variance_layer_{layer_label}.png", dpi=150, bbox_inches="tight")
        plt.close()

        analysis[layer_label] = {
            "ordinal_correlation": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in ordinal_results.items()},
            "pca_explained_variance": explained.tolist(),
        }

    return analysis


def analyze_cross_form_similarity(results, layer_indices):
    """Compare how similar French vs Belgian vs digit representations are for the same number."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cosine

    last_layer = max(layer_indices)
    layer_results = [r for r in results if r["layer"] == last_layer]

    # Group by number
    by_number = {}
    for r in layer_results:
        num = r["number"]
        if num not in by_number:
            by_number[num] = {}
        by_number[num][r["form"]] = r["hidden_state"]

    # For each number, compute pairwise cosine similarity between forms
    similarities = {"french_digit": [], "belgian_digit": [], "french_belgian": []}
    numbers_with_all = []
    vig_status = []

    for num in sorted(by_number.keys()):
        forms = by_number[num]
        if "french" in forms and "digit" in forms:
            sim = 1 - cosine(forms["french"], forms["digit"])
            similarities["french_digit"].append(sim)

        if "belgian" in forms and "digit" in forms:
            sim = 1 - cosine(forms["belgian"], forms["digit"])
            similarities["belgian_digit"].append(sim)

        if "french" in forms and "belgian" in forms:
            sim = 1 - cosine(forms["french"], forms["belgian"])
            similarities["french_belgian"].append(sim)
            numbers_with_all.append(num)
            vig_status.append(num >= 70)

    # Plot cross-form similarity
    if similarities["french_digit"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: French-Digit similarity by number
        ax = axes[0]
        all_numbers_0_99 = list(range(len(similarities["french_digit"])))
        ax.plot(all_numbers_0_99, similarities["french_digit"], 'b-', alpha=0.7, label="French-Digit")
        if similarities["belgian_digit"]:
            # Belgian only for 70-99
            belgian_x = list(range(70, 70 + len(similarities["belgian_digit"])))
            ax.plot(belgian_x, similarities["belgian_digit"], 'r-', alpha=0.7, label="Belgian-Digit")
        ax.axvspan(70, 99, alpha=0.1, color="red", label="Vigesimal range")
        ax.set_xlabel("Number")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Similarity Between Number Forms (Last Layer)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: French-Belgian similarity in vigesimal range
        if similarities["french_belgian"]:
            ax = axes[1]
            ax.bar(numbers_with_all, similarities["french_belgian"],
                   color=["red" if v else "blue" for v in vig_status])
            ax.set_xlabel("Number")
            ax.set_ylabel("Cosine Similarity")
            ax.set_title("French vs Belgian French Similarity")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "cross_form_similarity.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {PLOTS_DIR}/cross_form_similarity.png")

    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in similarities.items() if v}


def analyze_tokenization(tokenizer, numbers_data):
    """Analyze how French vs Belgian vs digit numbers are tokenized."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    token_counts = {"french": [], "belgian": [], "digit": []}
    numbers = []

    for entry in numbers_data:
        num = entry["number"]
        numbers.append(num)
        fr_tokens = tokenizer.encode(entry["french_word"], add_special_tokens=False)
        token_counts["french"].append(len(fr_tokens))
        token_counts["digit"].append(len(tokenizer.encode(str(num), add_special_tokens=False)))
        if "belgian_french_word" in entry:
            token_counts["belgian"].append(len(tokenizer.encode(entry["belgian_french_word"], add_special_tokens=False)))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(numbers, token_counts["french"], 'b-', alpha=0.7, label="French")
    if token_counts["belgian"]:
        ax.plot(numbers, token_counts["belgian"], 'r-', alpha=0.7, label="Belgian French")
    ax.plot(numbers, token_counts["digit"], 'g-', alpha=0.7, label="Digit")
    ax.axvspan(70, 99, alpha=0.1, color="red", label="Vigesimal range")
    ax.set_xlabel("Number")
    ax.set_ylabel("Token Count")
    ax.set_title("Tokenization Length by Number Form")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / "tokenization_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/tokenization_analysis.png")

    # Compute stats
    fr_vig = [token_counts["french"][i] for i in range(len(numbers)) if 70 <= numbers[i] <= 99]
    fr_dec = [token_counts["french"][i] for i in range(len(numbers)) if 20 <= numbers[i] <= 69]

    return {
        "french_vigesimal_mean_tokens": float(np.mean(fr_vig)) if fr_vig else 0,
        "french_decimal_mean_tokens": float(np.mean(fr_dec)) if fr_dec else 0,
        "french_max_tokens": max(token_counts["french"]),
        "digit_max_tokens": max(token_counts["digit"]),
    }


def main():
    print("=" * 60)
    print("EXPERIMENT 2: Representational Analysis of French Numbers")
    print("=" * 60)

    # Load data — focus on 0-99
    with open(DATA_DIR / "french_numbers_0_999.json") as f:
        all_numbers = json.load(f)
    numbers_0_99 = [n for n in all_numbers if n["number"] <= 99]

    # Load vigesimal subset for Belgian French info
    with open(DATA_DIR / "vigesimal_subset_70_99.json") as f:
        vigesimal = json.load(f)

    # Merge Belgian French into 0-99 data
    belgian_lookup = {v["number"]: v["belgian_french_word"] for v in vigesimal}
    for entry in numbers_0_99:
        if entry["number"] in belgian_lookup:
            entry["belgian_french_word"] = belgian_lookup[entry["number"]]

    # Load model
    tokenizer, model = load_model()

    # --- Tokenization Analysis ---
    print("\n[2a] Tokenization analysis...")
    tok_stats = analyze_tokenization(tokenizer, numbers_0_99)
    print(f"  French vigesimal mean tokens: {tok_stats['french_vigesimal_mean_tokens']:.1f}")
    print(f"  French decimal mean tokens: {tok_stats['french_decimal_mean_tokens']:.1f}")

    # --- Extract Representations ---
    print("\n[2b] Extracting hidden state representations...")
    representations, layer_indices = extract_representations(tokenizer, model, numbers_0_99)

    # --- Analyze Representations ---
    print("\n[2c] Analyzing representations...")
    rep_analysis = analyze_representations(representations, layer_indices)

    # --- Cross-Form Similarity ---
    print("\n[2d] Cross-form similarity analysis...")
    cross_sim = analyze_cross_form_similarity(representations, layer_indices)
    print(f"  French-Digit similarity: {cross_sim.get('french_digit', {}).get('mean', 0):.3f}")
    if "belgian_digit" in cross_sim:
        print(f"  Belgian-Digit similarity: {cross_sim['belgian_digit']['mean']:.3f}")
    if "french_belgian" in cross_sim:
        print(f"  French-Belgian similarity: {cross_sim['french_belgian']['mean']:.3f}")

    # Save analysis
    analysis_results = {
        "model": MODEL_NAME,
        "layer_indices": layer_indices,
        "tokenization": tok_stats,
        "representation_analysis": rep_analysis,
        "cross_form_similarity": cross_sim,
    }
    with open(RESULTS_DIR / "representation_results.json", "w") as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Results saved to results/representation_results.json")
    print(f"Plots saved to {PLOTS_DIR}/")
    print("=" * 60)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
