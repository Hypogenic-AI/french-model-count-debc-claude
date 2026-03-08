# Cloned Repositories

## Repo 1: multilingual-number-puzzles
- **URL:** https://github.com/antara-raaghavi/multilingual-number-puzzles
- **Paper:** "Investigating the interaction of linguistic and mathematical reasoning in LMs using multilingual number puzzles" (EMNLP 2025)
- **Purpose:** Tests LLMs on cross-linguistic numeral system puzzles; shows implicit operator inference failure
- **Location:** code/multilingual-number-puzzles/
- **Key files:** Problem sets, evaluation scripts, operator manipulation code
- **Notes:** Directly applicable methodology — French vigesimal numbers use exactly the implicit operators this code tests. Can adapt the operator-manipulation experimental framework for French-specific evaluation.

## Repo 2: base10
- **URL:** https://github.com/amitlevy/base10
- **Paper:** "Language Models Encode Numbers Using Digit Representations in Base 10" (arXiv 2410.11781)
- **Purpose:** Per-digit circular probes for number representations in LLMs
- **Location:** code/base10/
- **Key files:** Circular probe implementation, causal intervention code, digit probing pipeline
- **Notes:** Core probing tool. The circular probes can be applied to French number word representations to test whether vigesimal words map to the same base-10 digit structure as digit tokens. Supports Llama 3 8B and Mistral 7B.

## Repo 3: number_cookbook
- **URL:** https://github.com/GraphPKU/number_cookbook
- **Paper:** "Number Cookbook: Number Understanding of Language Models and How to Improve It" (ICLR 2025)
- **Purpose:** NUPA benchmark for comprehensive numerical understanding evaluation
- **Location:** code/number_cookbook/
- **Key files:** Benchmark generation code, evaluation metrics (digit match, exact match), training scripts
- **Notes:** Provides benchmark infrastructure that can be adapted for French number evaluation. The digit-match metric is particularly relevant for vigesimal error analysis.

## Repo 4: numllama
- **URL:** https://github.com/prompteus/numllama
- **Paper:** "Unravelling the Mechanisms of Manipulating Numbers in Language Models" (arXiv 2510.26285)
- **Purpose:** Sinusoidal probing of number representations across models and layers
- **Location:** code/numllama/
- **Key files:** Sinusoidal probe implementation, Fourier analysis, cross-model RSA, error-layer identification
- **Notes:** Provides the sinusoidal probing methodology to test universality of number representations across French vs. English contexts. Can identify which layers introduce errors when processing French number words.

## Recommended Additional Tools (not cloned)

### TransformerLens
- **URL:** https://github.com/TransformerLensOrg/TransformerLens
- **Purpose:** Mechanistic interpretability library for GPT-style models
- **Install:** `pip install transformer-lens`
- **Notes:** Primary tool for extracting activations from models processing French number tokens. Supports 50+ models including Llama, Mistral, GPT-2.

### nnsight
- **URL:** https://github.com/ndif-team/nnsight
- **Purpose:** Flexible interpretability for any PyTorch model
- **Install:** `pip install nnsight`
- **Notes:** Alternative to TransformerLens with broader model support. Good for activation patching and causal interventions.
