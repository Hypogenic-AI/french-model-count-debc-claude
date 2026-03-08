# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project: "How does the model count in French?" Resources include 12 papers, 1 custom dataset (with 5 sub-datasets), and 4 code repositories.

## Papers
Total papers downloaded: 12

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Mechanistic Interpretability of Large-Scale Counting in LLMs | Various | 2026 | papers/2601.02989_mechanistic_counting_llms.pdf | CountScope probing; counting saturates at ~10 items |
| Pre-trained LMs Learn Accurate Number Representations | Kadlcik et al. | 2025 | papers/2506.08966_number_representations_accurate.pdf | Sinusoidal probes decode numbers with near-perfect accuracy |
| Addition and Subtraction in Transformers | Various | 2024 | papers/2402.02619_addition_subtraction_transformers.pdf | Mechanistic analysis of arithmetic circuits |
| Mechanisms of Manipulating Numbers in LMs | Stefanik et al. | 2025 | papers/2510.26285_mechanisms_manipulating_numbers.pdf | Universal sinusoidal representations across models |
| Language Models Do Not Embed Numbers Continuously | Davies et al. | 2025 | papers/2510.08009_numbers_not_continuous.pdf | Per-digit circular representations dominate |
| Cross-Linguistic Numeral Puzzles | Bhattacharya et al. | 2025 | papers/2506.13886_crosslinguistic_numeral_puzzles.pdf | **Most relevant** — implicit operators are the bottleneck |
| Tokenization Counts | Singh, Strouse | 2024 | papers/2402.14903_tokenization_arithmetic.pdf | Tokenization direction affects arithmetic accuracy |
| Digit Representations in Base 10 | Levy, Geva | 2024 | papers/2410.11781_digit_representations_base10.pdf | Foundational base-10 circular probes |
| Number Cookbook (NUPA) | GraphPKU | 2024 | papers/2411.03766_number_cookbook.pdf | 41-task benchmark; digit-match metric |
| Fragile Number Sense | Rahman, Mishra | 2025 | papers/2509.06332_fragile_number_sense.pdf | LLM number sense is pattern-matching |
| What is a Number for an LLM? | Marjieh et al. | 2025 | papers/2502.01540_what_is_number_llm.pdf | String-number entanglement in representations |
| Mechanistic Evaluation of Transformers | Arora et al. | 2025 | papers/2505.15105_mechanistic_eval.pdf | Causal intervention methodology |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets: 1 (custom-generated, with 5 sub-datasets)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| French Numbers 0-999 | Custom | 1,000 entries | Probing, arithmetic, counting | datasets/french_numbers/ | Full mapping with vigesimal annotations |
| Vigesimal Subset 70-99 | Custom | 30 entries | Focused vigesimal analysis | datasets/french_numbers/ | French vs. Belgian French |
| Arithmetic Tasks | Custom | 1,000 tasks | Addition in French | datasets/french_numbers/ | Flags for vigesimal involvement |
| Counting Prompts | Custom | 48 tasks | Counting sequences | datasets/french_numbers/ | Tests boundary crossing |
| Comparison Tasks | Custom | 499 tasks | Number comparison | datasets/french_numbers/ | Tests magnitude understanding |

See datasets/README.md for detailed descriptions and loading instructions.

## Code Repositories
Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| multilingual-number-puzzles | github.com/antara-raaghavi/multilingual-number-puzzles | Cross-linguistic numeral evaluation | code/multilingual-number-puzzles/ | Directly applicable framework |
| base10 | github.com/amitlevy/base10 | Per-digit circular probes | code/base10/ | Core probing tool |
| number_cookbook | github.com/GraphPKU/number_cookbook | NUPA benchmark | code/number_cookbook/ | Adaptable benchmark |
| numllama | github.com/prompteus/numllama | Sinusoidal probing | code/numllama/ | Cross-model analysis |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Searched arXiv, Semantic Scholar, and Google Scholar for papers on: French counting + LLMs, number representation in transformers, mechanistic interpretability of arithmetic, cross-linguistic numeral systems, tokenization effects on arithmetic
2. Downloaded all directly relevant papers (12 total)
3. Deep-read 6 core papers, skimmed abstracts of remaining 6
4. Created custom French number dataset since no existing one was found
5. Cloned all 4 code repositories referenced in the papers

### Selection Criteria
- Papers selected for direct relevance to French counting system processing in LLMs
- Prioritized mechanistic interpretability approaches over purely behavioral evaluations
- Included both foundational work (base-10 probes, sinusoidal representations) and application-specific papers (cross-linguistic numerals)

### Challenges Encountered
- No existing French number dataset found — created comprehensive one
- No paper has specifically studied French counting in LLMs — this is the research gap
- The numllama repo exists but may have limited documentation

### Gaps and Workarounds
- **Missing:** Direct study of French number representations → Our experiment fills this gap
- **Missing:** French arithmetic benchmark → Created custom dataset with vigesimal annotations
- **Missing:** Belgian/Swiss French comparison → Included Belgian French variants in dataset for natural control condition

## Recommendations for Experiment Design

Based on gathered resources:

1. **Primary dataset(s):** Custom French Numbers 0-999 dataset with vigesimal/decimal annotations; vigesimal subset (70-99) for focused analysis. Belgian French variants as control.

2. **Baseline methods:**
   - Compare French word → number probing accuracy vs. English word → number vs. digit → number
   - Use sinusoidal probes (from numllama/2510.26285) and circular probes (from base10/2410.11781)
   - Compare vigesimal range (70-99) vs. decimal range (20-69) probe accuracy

3. **Evaluation metrics:**
   - Sinusoidal probe accuracy (number value recovery from hidden states)
   - Per-digit circular probe accuracy (base-10 digit recovery)
   - Exact match on arithmetic and counting tasks
   - RSA between French and English number representations
   - String-number entanglement scores (from 2502.01540 methodology)

4. **Code to adapt/reuse:**
   - `base10` repo: Circular probes, causal interventions
   - `numllama` repo: Sinusoidal probes, cross-layer analysis
   - `number_cookbook` repo: Benchmark evaluation framework
   - TransformerLens: Activation extraction from models processing French tokens

5. **Recommended models:**
   - Llama 3.1 8B (well-studied, multilingual)
   - Mistral 7B (strong French support, developed in France)
   - Qwen2.5 7B (additional comparison point)

6. **Key experiments:**
   - **Experiment 1:** Probe number representations when model processes French number words vs. digits vs. English words. Do vigesimal French numbers (70-99) show lower probe accuracy?
   - **Experiment 2:** Compare French vs. Belgian French representations. Does septante/nonante (decimal) produce cleaner representations than soixante-dix/quatre-vingt-dix (vigesimal)?
   - **Experiment 3:** Test arithmetic accuracy in French. Do tasks involving vigesimal numbers show more errors? Are errors digit-based or value-based?
   - **Experiment 4:** Counting sequence generation. Can models count correctly through vigesimal boundaries (69→70, 79→80, 89→90)?
   - **Experiment 5:** Causal interventions — patch vigesimal representations with decimal equivalents to test whether the model's internal representation is the bottleneck.
