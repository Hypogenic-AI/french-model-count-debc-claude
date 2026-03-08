"""
Experiment 1: Behavioral evaluation of LLM French number processing.
Tests GPT-4.1 on:
  - French word → number conversion
  - Number → French word conversion
  - Arithmetic in French
  - Counting sequences across vigesimal boundaries
"""

import json
import os
import random
import time
from pathlib import Path
from openai import OpenAI

random.seed(42)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4.1"

DATA_DIR = Path(__file__).parent.parent / "datasets" / "french_numbers"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_dataset():
    with open(DATA_DIR / "french_numbers_0_999.json") as f:
        return json.load(f)


def load_vigesimal_subset():
    with open(DATA_DIR / "vigesimal_subset_70_99.json") as f:
        return json.load(f)


def load_arithmetic():
    with open(DATA_DIR / "arithmetic_tasks_sample.json") as f:
        return json.load(f)


def load_counting():
    with open(DATA_DIR / "counting_prompts.json") as f:
        return json.load(f)


def call_gpt(prompt, system="You are a helpful assistant. Answer concisely with just the requested value, no explanation."):
    """Call GPT-4.1 with retry logic."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"


def experiment_french_to_number(numbers_subset):
    """Test: Given a French number word, what is the number?"""
    results = []
    for entry in numbers_subset:
        prompt = f"What number is '{entry['french_word']}' in French? Reply with just the number."
        response = call_gpt(prompt)
        try:
            predicted = int(response.replace(",", "").replace(".", "").strip())
        except ValueError:
            predicted = -1
        correct = predicted == entry["number"]
        results.append({
            "number": entry["number"],
            "french_word": entry["french_word"],
            "predicted": predicted,
            "correct": correct,
            "uses_vigesimal": entry["uses_vigesimal"],
            "category": entry["category"],
            "raw_response": response
        })
    return results


def experiment_number_to_french(numbers_subset):
    """Test: Given a number, what is the French word?"""
    results = []
    for entry in numbers_subset:
        prompt = f"Write the number {entry['number']} in French words. Reply with just the French word(s)."
        response = call_gpt(prompt)
        # Normalize comparison
        expected = entry["french_word"].lower().strip()
        got = response.lower().strip().rstrip(".")
        # Allow minor variations (hyphens vs spaces, etc.)
        correct = (got == expected or
                   got.replace("-", " ") == expected.replace("-", " ") or
                   got.replace(" ", "-") == expected.replace(" ", "-"))
        results.append({
            "number": entry["number"],
            "expected_french": entry["french_word"],
            "predicted_french": response,
            "correct": correct,
            "uses_vigesimal": entry["uses_vigesimal"],
            "category": entry["category"]
        })
    return results


def experiment_belgian_french(vigesimal_subset):
    """Test: Does the model know Belgian French number words?"""
    results = []
    for entry in vigesimal_subset:
        # Test Belgian French → number
        prompt = f"In Belgian French, the number word is '{entry['belgian_french_word']}'. What number is this? Reply with just the number."
        response = call_gpt(prompt)
        try:
            predicted = int(response.replace(",", "").replace(".", "").strip())
        except ValueError:
            predicted = -1
        correct = predicted == entry["number"]
        results.append({
            "number": entry["number"],
            "belgian_word": entry["belgian_french_word"],
            "french_word": entry["french_word"],
            "predicted": predicted,
            "correct": correct,
            "category": entry["category"],
            "raw_response": response
        })
    return results


def experiment_arithmetic(arithmetic_tasks):
    """Test arithmetic in French, comparing vigesimal vs decimal involvement."""
    results = []
    for task in arithmetic_tasks[:100]:  # First 100 tasks
        prompt = (
            f"Combien font {task['a_french']} plus {task['b_french']} ? "
            f"Répondez uniquement avec le nombre en chiffres."
        )
        response = call_gpt(prompt)
        try:
            predicted = int(response.replace(",", "").replace(".", "").strip())
        except ValueError:
            predicted = -1
        correct = predicted == task["result"]
        results.append({
            "a": task["a"],
            "b": task["b"],
            "expected": task["result"],
            "predicted": predicted,
            "correct": correct,
            "involves_vigesimal_input": task["involves_vigesimal_input"],
            "involves_vigesimal_output": task["involves_vigesimal_output"],
            "a_french": task["a_french"],
            "b_french": task["b_french"]
        })
    return results


def experiment_counting():
    """Test counting sequences across vigesimal boundaries."""
    boundaries = [
        {"start": 67, "end": 73, "label": "69→70 boundary"},
        {"start": 77, "end": 83, "label": "79→80 boundary"},
        {"start": 87, "end": 93, "label": "89→90 boundary"},
        {"start": 27, "end": 33, "label": "29→30 control"},
        {"start": 47, "end": 53, "label": "49→50 control"},
        {"start": 57, "end": 63, "label": "59→60 control"},
    ]
    results = []
    for b in boundaries:
        prompt = (
            f"Comptez en français de {b['start']} à {b['end']}. "
            f"Écrivez chaque nombre en toutes lettres, un par ligne."
        )
        response = call_gpt(prompt, system="Vous êtes un assistant utile. Répondez uniquement avec la liste des nombres en lettres.")
        # Parse the response into individual number words
        lines = [l.strip().lstrip("0123456789.-) ").strip() for l in response.strip().split("\n") if l.strip()]
        results.append({
            "start": b["start"],
            "end": b["end"],
            "label": b["label"],
            "is_vigesimal_boundary": "control" not in b["label"],
            "response": response,
            "parsed_lines": lines,
            "expected_count": b["end"] - b["start"] + 1,
            "actual_count": len(lines)
        })
    return results


def compute_summary_stats(results, label):
    """Compute accuracy stats grouped by vigesimal status."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    vigesimal = [r for r in results if r.get("uses_vigesimal", False)]
    decimal = [r for r in results if not r.get("uses_vigesimal", False)]

    vig_correct = sum(1 for r in vigesimal if r["correct"]) if vigesimal else 0
    dec_correct = sum(1 for r in decimal if r["correct"]) if decimal else 0

    stats = {
        "experiment": label,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
        "vigesimal_total": len(vigesimal),
        "vigesimal_correct": vig_correct,
        "vigesimal_accuracy": vig_correct / len(vigesimal) if vigesimal else None,
        "decimal_total": len(decimal),
        "decimal_correct": dec_correct,
        "decimal_accuracy": dec_correct / len(decimal) if decimal else None,
    }
    return stats


def main():
    print("=" * 60)
    print("EXPERIMENT 1: Behavioral Evaluation of French Number Processing")
    print("=" * 60)

    # Load data
    all_numbers = load_dataset()
    vigesimal_subset = load_vigesimal_subset()
    arithmetic = load_arithmetic()

    # Select test numbers: 0-99 for focused analysis
    test_numbers = [n for n in all_numbers if n["number"] <= 99]

    all_results = {}
    all_stats = {}

    # --- Experiment 1a: French → Number ---
    print("\n[1a] French word → Number conversion (0-99)...")
    fr_to_num = experiment_french_to_number(test_numbers)
    stats = compute_summary_stats(fr_to_num, "french_to_number")
    all_results["french_to_number"] = fr_to_num
    all_stats["french_to_number"] = stats
    print(f"  Overall: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    print(f"  Decimal (0-69): {stats['decimal_accuracy']:.1%} ({stats['decimal_correct']}/{stats['decimal_total']})")
    print(f"  Vigesimal (70-99): {stats['vigesimal_accuracy']:.1%} ({stats['vigesimal_correct']}/{stats['vigesimal_total']})")

    # --- Experiment 1b: Number → French ---
    print("\n[1b] Number → French word conversion (0-99)...")
    num_to_fr = experiment_number_to_french(test_numbers)
    stats = compute_summary_stats(num_to_fr, "number_to_french")
    all_results["number_to_french"] = num_to_fr
    all_stats["number_to_french"] = stats
    print(f"  Overall: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    print(f"  Decimal (0-69): {stats['decimal_accuracy']:.1%} ({stats['decimal_correct']}/{stats['decimal_total']})")
    print(f"  Vigesimal (70-99): {stats['vigesimal_accuracy']:.1%} ({stats['vigesimal_correct']}/{stats['vigesimal_total']})")

    # --- Experiment 1c: Belgian French → Number ---
    print("\n[1c] Belgian French → Number conversion (70-99)...")
    belgian = experiment_belgian_french(vigesimal_subset)
    belgian_correct = sum(1 for r in belgian if r["correct"])
    belgian_stats = {
        "experiment": "belgian_french_to_number",
        "total": len(belgian),
        "correct": belgian_correct,
        "accuracy": belgian_correct / len(belgian),
    }
    all_results["belgian_french"] = belgian
    all_stats["belgian_french"] = belgian_stats
    print(f"  Belgian French accuracy: {belgian_stats['accuracy']:.1%} ({belgian_correct}/{len(belgian)})")

    # --- Experiment 1d: Arithmetic in French ---
    print("\n[1d] Arithmetic in French (100 tasks)...")
    arith = experiment_arithmetic(arithmetic)
    # Group by vigesimal involvement
    vig_input = [r for r in arith if r["involves_vigesimal_input"]]
    vig_output = [r for r in arith if r["involves_vigesimal_output"]]
    no_vig = [r for r in arith if not r["involves_vigesimal_input"] and not r["involves_vigesimal_output"]]

    arith_stats = {
        "experiment": "arithmetic",
        "total": len(arith),
        "correct": sum(1 for r in arith if r["correct"]),
        "accuracy": sum(1 for r in arith if r["correct"]) / len(arith),
        "vigesimal_input_total": len(vig_input),
        "vigesimal_input_correct": sum(1 for r in vig_input if r["correct"]),
        "vigesimal_input_accuracy": sum(1 for r in vig_input if r["correct"]) / len(vig_input) if vig_input else None,
        "vigesimal_output_total": len(vig_output),
        "vigesimal_output_correct": sum(1 for r in vig_output if r["correct"]),
        "vigesimal_output_accuracy": sum(1 for r in vig_output if r["correct"]) / len(vig_output) if vig_output else None,
        "no_vigesimal_total": len(no_vig),
        "no_vigesimal_correct": sum(1 for r in no_vig if r["correct"]),
        "no_vigesimal_accuracy": sum(1 for r in no_vig if r["correct"]) / len(no_vig) if no_vig else None,
    }
    all_results["arithmetic"] = arith
    all_stats["arithmetic"] = arith_stats
    print(f"  Overall: {arith_stats['accuracy']:.1%}")
    if no_vig:
        print(f"  No vigesimal: {arith_stats['no_vigesimal_accuracy']:.1%}")
    if vig_input:
        print(f"  Vigesimal input: {arith_stats['vigesimal_input_accuracy']:.1%}")
    if vig_output:
        print(f"  Vigesimal output: {arith_stats['vigesimal_output_accuracy']:.1%}")

    # --- Experiment 1e: Counting Sequences ---
    print("\n[1e] Counting sequences across boundaries...")
    counting = experiment_counting()
    all_results["counting"] = counting
    for c in counting:
        print(f"  {c['label']}: {c['actual_count']} items (expected {c['expected_count']})")

    # Save all results
    with open(RESULTS_DIR / "behavioral_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / "behavioral_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Results saved to results/behavioral_results.json")
    print("Stats saved to results/behavioral_stats.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
