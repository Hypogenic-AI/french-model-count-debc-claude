# Literature Review: How Does the Model Count in French?

## Research Area Overview

This review surveys the intersection of three active research areas: (1) how LLMs internally represent numbers, (2) mechanistic interpretability of arithmetic and counting, and (3) cross-linguistic numeral systems in language models. The French counting system is a compelling test case because it combines standard decimal counting (1-69) with a vigesimal (base-20) system for 70-99 (e.g., "quatre-vingt-dix-sept" = 4×20+10+7 = 97), requiring implicit multiplication and addition operations embedded in language.

## Key Papers

### Paper 1: Investigating the Interaction of Linguistic and Mathematical Reasoning Using Multilingual Number Puzzles
- **Authors:** Bhattacharya, Papadimitriou, Davidson, Alvarez-Melis (Harvard)
- **Year:** 2025 (EMNLP)
- **Source:** arXiv:2506.13886
- **Key Contribution:** Demonstrates that LLMs fail at cross-linguistic numeral puzzles not because of mathematical difficulty but because they cannot infer **implicit compositional operators** (addition, multiplication) from numeral structure. Models succeed when operators are made explicit with familiar symbols (+, ×).
- **Methodology:** 10 Linguistics Olympiad problems across 10 languages (Drehu, Georgian, Yoruba, etc.), testing o1-mini and DeepSeek-R1 under 4 operator conditions (implicit, explicit+familiar, explicit+unfamiliar symbol, explicit+unfamiliar word).
- **Datasets:** LingOly and Linguini benchmarks (filtered for number systems).
- **Results:** Familiar explicit operators (+, ×) yield ceiling performance; implicit operators lead to failure. Language name context boosts performance by ~30%, suggesting memorization over reasoning.
- **Code:** https://github.com/antara-raaghavi/multilingual-number-puzzles
- **Relevance:** Directly applicable — French vigesimal numbers contain exactly the kind of implicit operators (60+10=70, 4×20=80) that this paper shows models cannot infer. The authors explicitly cite French as an example vigesimal system analogous to their test languages.

### Paper 2: Unravelling the Mechanisms of Manipulating Numbers in Language Models
- **Authors:** Stefanik, Kadlcik, Mickus, Spiegel, Kuchar (Masaryk U., U. Helsinki)
- **Year:** 2025
- **Source:** arXiv:2510.26285
- **Key Contribution:** Shows that LLMs converge on **sinusoidal representations** of numbers that are universal across models, layers, and contexts. Identifies specific layers where errors are introduced and shows they can be mitigated.
- **Methodology:** Sinusoidal probes, RSA, Fourier decomposition across 8 LLMs (Llama 3, OLMo 2, Phi 4 families). TunedLens comparisons. Natural language contexts across arithmetic, temporal, medical, and culinary domains.
- **Results:** Sinusoidal probes consistently outperform linear/log probes. Cross-model Fourier frequency IoU = 1.0 for top-63 frequencies. Universal probes achieve 95-100% cross-layer accuracy. Up to 94.4% of surfaced errors correspond to correct internal computations that failed to surface.
- **Code:** https://github.com/prompteus/numllama
- **Relevance:** Establishes the probing methodology needed to investigate French number representations. The finding that representations are universal raises the question: does this universality hold for French number words, or does the vigesimal structure create distinct patterns?

### Paper 3: Language Models Encode Numbers Using Digit Representations in Base 10
- **Authors:** Levy, Geva (Tel Aviv University)
- **Year:** 2024
- **Source:** arXiv:2410.11781
- **Key Contribution:** Demonstrates that LLMs use **per-digit circular representations in base 10**, not continuous scalar encodings. Probes can recover individual digit values but not full number values.
- **Methodology:** Circular probes mapping digit values to unit circle positions, tested on Llama 3 8B and Mistral 7B. Causal interventions (flipping digit representations during inference).
- **Results:** Base-10 digit probes achieve 91-96% accuracy; all other bases (2-14) below 0.2. ~80% of arithmetic errors affect only a single digit. Probes partially generalize to English word forms (68.6% peak).
- **Code:** https://github.com/amitlevy/base10
- **Relevance:** Critical finding — models represent numbers in base 10 internally, even when surface forms use different systems. French vigesimal words must be mapped to base-10 digit representations, creating a mismatch between linguistic form and internal representation.

### Paper 4: Language Models Do Not Embed Numbers Continuously
- **Authors:** Davies et al.
- **Year:** 2025
- **Source:** arXiv:2510.08009
- **Key Contribution:** Shows that embedding models' number representations are far from clean continuous scalar encodings. String artifacts, digit patterns, and tokenization effects dominate the embedding space.
- **Methodology:** PCA analysis of number embeddings from OpenAI, Google Gemini, and Voyage AI models. Linear reconstruction, explained variance analysis.
- **Results:** R² ≥ 0.95 for value reconstruction, but first PC explains ≤40% of variance. Negative numbers and large magnitudes degrade representations further.
- **Relevance:** Predicts that French number words would produce even more complex embedding structures due to the additional string-level variation in vigesimal forms.

### Paper 5: Pre-trained Language Models Learn Remarkably Accurate Representations of Numbers
- **Authors:** Kadlcik, Stefanik, Mickus, Spiegel, Kuchar
- **Year:** 2025 (EMNLP)
- **Source:** arXiv:2506.08966
- **Key Contribution:** Novel sinusoidal-basis probe achieves near-perfect accuracy in decoding numeric values from LLM embeddings, overturning prior findings of imprecise representations.
- **Methodology:** Sinusoidal probe (f_sin) using Fourier basis as inductive bias, tested on Llama 3, Phi 4, OLMo 2 (1B-72B parameters).
- **Relevance:** Provides the probing tool to test whether French number-word embeddings encode magnitude as precisely as digit tokens.

### Paper 6: Mechanistic Interpretability of Large-Scale Counting in LLMs
- **Authors:** (Various)
- **Year:** 2026
- **Source:** arXiv:2601.02989
- **Key Contribution:** Shows counting in LLMs is a layerwise process that saturates beyond ~10 items due to depth constraints. Proposes System-2 decomposition strategy.
- **Methodology:** CountScope probing, attention analysis, causal mediation analysis on Qwen2.5 7B, Llama 3 8B, Gemma 3 27B.
- **Results:** Numbers encoded with logarithmic compression (mental number line). Specific attention heads mediate count information transfer. Systematic biases toward round numbers.
- **Relevance:** French vigesimal number words are longer (more tokens) than English equivalents, potentially consuming more transformer depth for the same count.

### Paper 7: Number Cookbook: Number Understanding of Language Models and How to Improve It
- **Authors:** (GraphPKU)
- **Year:** 2024 (ICLR 2025)
- **Source:** arXiv:2411.03766
- **Key Contribution:** NUPA benchmark with 4 representations × 17 tasks = 41 combinations. Shows performance degrades sharply with number length and that one-digit tokenizers outperform multi-digit.
- **Code:** https://github.com/GraphPKU/number_cookbook
- **Relevance:** Benchmark framework and metrics (digit match) adaptable for French number evaluation. Tokenization findings directly relevant since French words tokenize differently from digits.

### Paper 8: Tokenization Counts: The Impact of Tokenization on Arithmetic in Frontier LLMs
- **Authors:** Singh, Strouse (UCL, Google DeepMind)
- **Year:** 2024
- **Source:** arXiv:2402.14903
- **Key Contribution:** Shows tokenization direction (L→R vs R→L) significantly impacts arithmetic accuracy, with R→L improving by up to 20%.
- **Relevance:** French number words tokenize very differently from Arabic digits, creating unique tokenization-dependent biases.

### Paper 9: What is a Number, That a Large Language Model May Know It?
- **Authors:** Marjieh, Veselovsky, Griffiths, Sucholutsky (Princeton, NYU)
- **Year:** 2025
- **Source:** arXiv:2502.01540
- **Key Contribution:** LLMs learn entangled representations blending string-level (edit distance) and numerical (log-linear magnitude) properties. Context reduces but cannot eliminate this entanglement.
- **Relevance:** Very high — French vigesimal words create string similarities among numerically distant numbers (e.g., "quatre-vingt-un" [81] similar to "quatre-vingt-onze" [91]), predicting distinctive entanglement patterns.

### Paper 10: A Fragile Number Sense
- **Authors:** Rahman, Mishra (Stanford, SLAC)
- **Year:** 2025
- **Source:** arXiv:2509.06332
- **Key Contribution:** LLM numerical reasoning is fragile — models succeed at deterministic tasks but fail at combinatorial puzzles, revealing pattern-matching over genuine number sense.
- **Relevance:** French counting irregularities may expose additional fragilities not visible in English.

## Common Methodologies

- **Probing (Linear/Sinusoidal/Circular):** Used in Papers 2, 3, 5 to decode number values from hidden states. Sinusoidal probes outperform linear ones.
- **Causal Interventions:** Papers 3, 6 use activation patching to establish causal relationships between representations and outputs.
- **Similarity Analysis:** Paper 9 uses pairwise similarity judgments and RSA to characterize representation structure.
- **Behavioral Evaluation:** Papers 1, 7, 8, 10 evaluate model outputs on numerical tasks.

## Standard Baselines
- **Models:** Llama 3/3.1 (8B, 70B), Mistral 7B, GPT-4o, Qwen2.5 7B, OLMo 2, Phi 4
- **Tasks:** Addition, comparison, counting, digit extraction, number-to-word conversion
- **Metrics:** Exact match, digit match, classification accuracy, probe R²

## Evaluation Metrics in the Literature
- **Exact Match:** Binary correct/incorrect on outputs
- **Digit Match:** Per-digit accuracy (from Number Cookbook)
- **Probe Accuracy:** Classification accuracy of probes on hidden states
- **RSA Scores:** Representational similarity between models/layers
- **Fourier IoU:** Overlap of dominant frequencies in number embeddings

## Datasets in the Literature
- **LingOly/Linguini:** Cross-linguistic number system puzzles (Paper 1)
- **NUPA Benchmark:** 41 numerical task combinations (Paper 7)
- **Synthetic counting tasks:** Lists of items for counting (Paper 6)
- **MetaMathQA, DROP, AQuA-RAT:** Arithmetic in context (Paper 2)
- **No existing French number dataset** was found — we created one for this project.

## Gaps and Opportunities

1. **No study has probed French number representations specifically.** All probing work (Papers 2, 3, 5) uses English contexts or digit tokens. The vigesimal system is an untested case.
2. **Implicit operator inference is unsolved.** Paper 1 shows models cannot infer implicit compositional operations — exactly what French numbers require.
3. **The base-10 internal representation creates a mismatch.** Paper 3 shows models use base-10 circular encodings internally. French vigesimal words must be mapped to base-10, but no one has studied this mapping.
4. **String-number entanglement in French is unexplored.** Paper 9's methodology could reveal how French vigesimal word structure creates unique entanglement patterns.
5. **Belgian/Swiss French provides a natural control.** Belgian French uses septante (70), huitante (80), nonante (90) — standard decimal words. Comparing French vs. Belgian French representations would isolate the effect of vigesimal structure.

## Recommendations for Our Experiment

Based on the literature:

- **Primary approach:** Use sinusoidal and circular probes (from Papers 2, 3, 5) to analyze how models represent French numbers internally, comparing digit tokens, English words, standard French words, and Belgian French words.
- **Recommended datasets:** Our custom French number dataset (0-999) with vigesimal/decimal annotations, arithmetic tasks, counting prompts, and comparison tasks.
- **Recommended baselines:** Compare French vs. English vs. Belgian French vs. digit representations at each layer.
- **Recommended metrics:** Probe accuracy (sinusoidal and circular), digit match for arithmetic tasks, RSA between French and English number representations.
- **Recommended models:** Llama 3.1 8B (well-studied, multilingual), Mistral 7B (strong French language support), Qwen2.5 7B.
- **Tools:** TransformerLens or nnsight for activation extraction, circular probes from base10 repo.
- **Key experiment:** Test whether vigesimal French numbers (70-99) show lower probe accuracy, different dominant Fourier frequencies, or higher string-number entanglement compared to decimal-range numbers (1-69) and Belgian French equivalents.
