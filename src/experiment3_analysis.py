"""
Experiment 3: Statistical analysis and comprehensive visualization.
Reads results from experiments 1 & 2 and produces final analysis.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_results():
    behavioral = {}
    behavioral_stats = {}
    representation = {}

    try:
        with open(RESULTS_DIR / "behavioral_results.json") as f:
            behavioral = json.load(f)
        with open(RESULTS_DIR / "behavioral_stats.json") as f:
            behavioral_stats = json.load(f)
    except FileNotFoundError:
        print("Warning: behavioral results not found")

    try:
        with open(RESULTS_DIR / "representation_results.json") as f:
            representation = json.load(f)
    except FileNotFoundError:
        print("Warning: representation results not found")

    return behavioral, behavioral_stats, representation


def analyze_behavioral(behavioral, behavioral_stats):
    """Statistical analysis of behavioral results."""
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS: Behavioral Experiments")
    print("=" * 60)

    analysis = {}

    # --- French to Number: Chi-squared test ---
    if "french_to_number" in behavioral:
        results = behavioral["french_to_number"]
        vig = [r for r in results if r["uses_vigesimal"]]
        dec = [r for r in results if not r["uses_vigesimal"]]

        vig_correct = sum(1 for r in vig if r["correct"])
        dec_correct = sum(1 for r in dec if r["correct"])

        # 2x2 contingency table
        table = [
            [dec_correct, len(dec) - dec_correct],
            [vig_correct, len(vig) - vig_correct]
        ]
        if min(table[0][1], table[1][1]) > 0:  # Only if there are errors
            chi2, p, dof, expected = stats.chi2_contingency(table)
            # Cohen's h for effect size
            p1 = dec_correct / len(dec)
            p2 = vig_correct / len(vig)
            h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
            analysis["french_to_number"] = {
                "decimal_accuracy": p1,
                "vigesimal_accuracy": p2,
                "chi2": float(chi2),
                "p_value": float(p),
                "cohens_h": float(h),
                "significant": p < 0.05
            }
            print(f"\n  French→Number:")
            print(f"    Decimal accuracy: {p1:.1%}")
            print(f"    Vigesimal accuracy: {p2:.1%}")
            print(f"    χ² = {chi2:.3f}, p = {p:.4f}")
            print(f"    Cohen's h = {h:.3f}")
            print(f"    {'SIGNIFICANT' if p < 0.05 else 'Not significant'}")
        else:
            analysis["french_to_number"] = {
                "decimal_accuracy": dec_correct / len(dec),
                "vigesimal_accuracy": vig_correct / len(vig),
                "note": "Perfect or near-perfect accuracy, no statistical test needed"
            }
            print(f"\n  French→Number: Near-perfect accuracy for both groups")

    # --- Number to French ---
    if "number_to_french" in behavioral:
        results = behavioral["number_to_french"]
        vig = [r for r in results if r["uses_vigesimal"]]
        dec = [r for r in results if not r["uses_vigesimal"]]

        vig_correct = sum(1 for r in vig if r["correct"])
        dec_correct = sum(1 for r in dec if r["correct"])

        p1 = dec_correct / len(dec)
        p2 = vig_correct / len(vig)

        table = [
            [dec_correct, len(dec) - dec_correct],
            [vig_correct, len(vig) - vig_correct]
        ]
        if min(table[0][1], table[1][1]) > 0:
            chi2, p, dof, expected = stats.chi2_contingency(table)
            h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
            analysis["number_to_french"] = {
                "decimal_accuracy": p1,
                "vigesimal_accuracy": p2,
                "chi2": float(chi2),
                "p_value": float(p),
                "cohens_h": float(h),
                "significant": p < 0.05
            }
            print(f"\n  Number→French:")
            print(f"    Decimal accuracy: {p1:.1%}")
            print(f"    Vigesimal accuracy: {p2:.1%}")
            print(f"    χ² = {chi2:.3f}, p = {p:.4f}")
        else:
            analysis["number_to_french"] = {
                "decimal_accuracy": p1,
                "vigesimal_accuracy": p2,
                "note": "Near-perfect accuracy"
            }
            print(f"\n  Number→French: Near-perfect accuracy for both groups")

        # Error analysis for number_to_french
        errors = [r for r in results if not r["correct"]]
        if errors:
            print(f"\n  Errors in Number→French ({len(errors)} total):")
            for e in errors[:10]:
                print(f"    {e['number']}: expected '{e['expected_french']}', got '{e['predicted_french']}'")

    # --- Arithmetic ---
    if "arithmetic" in behavioral:
        results = behavioral["arithmetic"]
        vig_in = [r for r in results if r["involves_vigesimal_input"]]
        vig_out = [r for r in results if r["involves_vigesimal_output"]]
        no_vig = [r for r in results if not r["involves_vigesimal_input"] and not r["involves_vigesimal_output"]]

        print(f"\n  Arithmetic:")
        print(f"    Overall: {sum(1 for r in results if r['correct'])}/{len(results)}")
        if no_vig:
            print(f"    No vigesimal: {sum(1 for r in no_vig if r['correct'])}/{len(no_vig)}")
        if vig_in:
            print(f"    Vigesimal input: {sum(1 for r in vig_in if r['correct'])}/{len(vig_in)}")
        if vig_out:
            print(f"    Vigesimal output: {sum(1 for r in vig_out if r['correct'])}/{len(vig_out)}")

        # Show arithmetic errors
        arith_errors = [r for r in results if not r["correct"]]
        if arith_errors:
            print(f"\n  Arithmetic errors ({len(arith_errors)} total):")
            for e in arith_errors[:10]:
                print(f"    {e['a_french']} + {e['b_french']} = {e['expected']} (got {e['predicted']})")

    return analysis


def create_summary_plots(behavioral, behavioral_stats):
    """Create summary visualization plots."""
    print("\n  Creating summary plots...")

    # --- Plot 1: Accuracy comparison bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Conversion accuracy
    if "french_to_number" in behavioral_stats and "number_to_french" in behavioral_stats:
        ax = axes[0]
        experiments = ["French→Number", "Number→French"]
        decimal_acc = [
            behavioral_stats["french_to_number"].get("decimal_accuracy", 0),
            behavioral_stats["number_to_french"].get("decimal_accuracy", 0),
        ]
        vigesimal_acc = [
            behavioral_stats["french_to_number"].get("vigesimal_accuracy", 0),
            behavioral_stats["number_to_french"].get("vigesimal_accuracy", 0),
        ]

        x = np.arange(len(experiments))
        width = 0.35
        ax.bar(x - width/2, decimal_acc, width, label="Decimal (0-69)", color="steelblue")
        ax.bar(x + width/2, vigesimal_acc, width, label="Vigesimal (70-99)", color="indianred")
        ax.set_ylabel("Accuracy")
        ax.set_title("Conversion Task Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(experiments)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

    # Arithmetic accuracy
    if "arithmetic" in behavioral_stats:
        ax = axes[1]
        arith = behavioral_stats["arithmetic"]
        labels = []
        values = []
        if arith.get("no_vigesimal_accuracy") is not None:
            labels.append("No vigesimal")
            values.append(arith["no_vigesimal_accuracy"])
        if arith.get("vigesimal_input_accuracy") is not None:
            labels.append("Vigesimal\ninput")
            values.append(arith["vigesimal_input_accuracy"])
        if arith.get("vigesimal_output_accuracy") is not None:
            labels.append("Vigesimal\noutput")
            values.append(arith["vigesimal_output_accuracy"])

        colors = ["steelblue", "indianred", "coral"]
        ax.bar(labels, values, color=colors[:len(labels)])
        ax.set_ylabel("Accuracy")
        ax.set_title("Arithmetic Accuracy by Vigesimal Involvement")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

    # Belgian French comparison
    if "french_to_number" in behavioral_stats and "belgian_french" in behavioral_stats:
        ax = axes[2]
        fr_vig_acc = behavioral_stats["french_to_number"].get("vigesimal_accuracy", 0)
        be_acc = behavioral_stats["belgian_french"]["accuracy"]

        ax.bar(["Standard French\n(vigesimal)", "Belgian French\n(decimal)"],
               [fr_vig_acc, be_acc],
               color=["indianred", "steelblue"])
        ax.set_ylabel("Accuracy")
        ax.set_title("French vs Belgian French (70-99)")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "behavioral_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/behavioral_summary.png")

    # --- Plot 2: Per-number accuracy heatmap ---
    if "french_to_number" in behavioral:
        results = behavioral["french_to_number"]
        fig, ax = plt.subplots(figsize=(12, 3))

        numbers = [r["number"] for r in results]
        correct = [1 if r["correct"] else 0 for r in results]

        # Create a 10x10 grid for 0-99
        grid = np.full((10, 10), np.nan)
        for n, c in zip(numbers, correct):
            if n < 100:
                grid[n // 10, n % 10] = c

        im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xlabel("Ones digit")
        ax.set_ylabel("Tens digit")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_yticklabels([f"{i}0s" for i in range(10)])

        # Add red box around vigesimal range
        rect = plt.Rectangle((-.5, 6.5), 10, 3, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.set_title("French→Number Accuracy (red border = vigesimal range 70-99)")
        plt.colorbar(im, ax=ax, label="Correct")
        plt.savefig(PLOTS_DIR / "accuracy_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {PLOTS_DIR}/accuracy_heatmap.png")

    # --- Plot 3: Error analysis ---
    if "number_to_french" in behavioral:
        results = behavioral["number_to_french"]
        errors = [r for r in results if not r["correct"]]
        if errors:
            error_numbers = [e["number"] for e in errors]
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.hist(error_numbers, bins=range(0, 101, 5), color="indianred", alpha=0.7, edgecolor="black")
            ax.axvspan(70, 99, alpha=0.1, color="red", label="Vigesimal range")
            ax.set_xlabel("Number")
            ax.set_ylabel("Error Count")
            ax.set_title("Distribution of Number→French Errors")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.savefig(PLOTS_DIR / "error_distribution.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {PLOTS_DIR}/error_distribution.png")


def main():
    behavioral, behavioral_stats, representation = load_results()

    # Statistical analysis
    stat_analysis = analyze_behavioral(behavioral, behavioral_stats)

    # Create plots
    create_summary_plots(behavioral, behavioral_stats)

    # Counting analysis
    if "counting" in behavioral:
        print("\n  Counting sequence analysis:")
        for c in behavioral["counting"]:
            status = "OK" if c["actual_count"] == c["expected_count"] else "MISMATCH"
            print(f"    {c['label']}: {status} ({c['actual_count']}/{c['expected_count']} items)")
            if c["parsed_lines"]:
                print(f"      Response: {', '.join(c['parsed_lines'][:5])}...")

    # Save final analysis
    final = {
        "behavioral_stats": behavioral_stats,
        "statistical_tests": stat_analysis,
        "representation_analysis": representation,
    }
    with open(RESULTS_DIR / "final_analysis.json", "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n  Final analysis saved to {RESULTS_DIR}/final_analysis.json")


if __name__ == "__main__":
    main()
