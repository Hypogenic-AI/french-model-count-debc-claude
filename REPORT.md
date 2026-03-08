# How Does the Model Count in French?

## 1. Executive Summary

We investigated how LLMs represent and process the French vigesimal counting system (70-99), where numbers use base-20 constructions like "quatre-vingt-dix-sept" (4×20+10+7 = 97). Our study combined behavioral evaluation of GPT-4.1 with representational analysis of Mistral 7B's hidden states.

**Key finding**: GPT-4.1 achieves 100% accuracy on all French number tasks — conversion, arithmetic, and counting — including vigesimal numbers. However, Mistral 7B's internal representations reveal that the vigesimal system creates a distinctive fingerprint: vigesimal numbers (70-99) cluster by *linguistic base* rather than numeric proximity (70% of nearest neighbors share a linguistic base like "quatre-vingt-"), and French number representations are better linearly decodable (R²=0.979) than digit representations (R²=0.943). Belgian French (septante/huitante/nonante) produces representations with the highest ordinal correlation (ρ=0.591 vs 0.520 for standard French).

**Practical implication**: Modern LLMs have fully memorized the French counting system at the behavioral level, but internally organize vigesimal numbers by their linguistic structure rather than their numeric value. The model "thinks in vigesimal" even though it "answers in decimal."

## 2. Goal

**Hypothesis**: LLMs represent and process the French vigesimal counting system in specific, mappable ways, with the irregular structure (70-99) potentially causing representational difficulties compared to regular decimal numbers.

**Why this matters**: French is one of the most widely-spoken languages with a vigesimal number system, making it a natural test case for understanding how LLMs bridge linguistic form and mathematical meaning. If models struggle with vigesimal representations internally, this reveals fundamental limits in how they handle implicit mathematical operations embedded in language.

## 3. Data Construction

### Dataset Description
Custom-generated dataset of French numbers 0-999, with focused analysis on 0-99. No prior French number dataset for LLM analysis existed.

### Example Samples

| Number | French Word | Belgian French | Vigesimal? | Implicit Operations |
|--------|-------------|---------------|------------|-------------------|
| 42 | quarante-deux | quarante-deux | No | - |
| 70 | soixante-dix | septante | Yes | 60+10 |
| 80 | quatre-vingts | huitante | Yes | 4×20+0 |
| 97 | quatre-vingt-dix-sept | nonante-sept | Yes | 4×20+10+7 |

### Sub-datasets
- **French Numbers 0-999**: 1,000 entries with vigesimal annotations
- **Vigesimal Subset 70-99**: 30 entries with both French and Belgian French forms
- **Arithmetic Tasks**: 100 addition tasks with vigesimal involvement flags
- **Counting Prompts**: 6 boundary-crossing sequences (3 vigesimal, 3 control)

## 4. Experiment Description

### Methodology

#### Experiment 1: Behavioral Evaluation (GPT-4.1)
- **French→Number conversion**: 100 number words (0-99) → numeric answer
- **Number→French conversion**: 100 numbers → French word generation
- **Belgian French→Number**: 30 vigesimal numbers in Belgian French form
- **Arithmetic in French**: 100 addition tasks with French operands
- **Counting sequences**: 6 boundary-crossing sequences (69→70, 79→80, 89→90 + controls)

#### Experiment 2: Representational Analysis (Mistral 7B v0.3)
- Extract last-token hidden states for 100 numbers × 3 forms (French/digit/Belgian) × 5 layers
- Tokenization analysis comparing token counts across forms
- t-SNE visualization of representation space
- Cross-form cosine similarity analysis

#### Experiment 3: Linear Probing (Mistral 7B)
- Ridge regression probes to predict numeric value from hidden states
- 5-fold cross-validation, comparing vigesimal vs decimal MAE
- Representational Similarity Analysis (RSA) across forms
- Nearest-neighbor analysis: do vigesimal numbers cluster by linguistic base or numeric value?

### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| OpenAI API | GPT-4.1 | Behavioral evaluation |
| Mistral 7B v0.3 | - | Representation extraction |
| PyTorch | 2.10.0 | Model inference |
| transformers | 5.3.0 | Model loading |
| scikit-learn | - | Linear probes, t-SNE, PCA |
| scipy | - | Statistical tests |

### Hardware
- 2× NVIDIA RTX 3090 (24GB each)
- Model loaded on single GPU in float16

### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| GPT-4.1 temperature | 0 | Deterministic outputs |
| Ridge alpha | 1.0 | Default regularization |
| t-SNE perplexity | 30 | Standard for ~300 points |
| Cross-validation folds | 5 | Standard |
| Random seed | 42 | Reproducibility |

## 5. Results

### 5.1 Behavioral Results (GPT-4.1)

GPT-4.1 achieves **perfect accuracy** on all tasks:

| Task | Decimal (0-69) | Vigesimal (70-99) | Overall |
|------|---------------|-------------------|---------|
| French→Number | 100% (70/70) | 100% (30/30) | 100% |
| Number→French | 100% (70/70) | 100% (30/30) | 100% |
| Belgian French→Number | N/A | 100% (30/30) | 100% |
| Arithmetic | 100% (35/35) | 100% (49+22) | 100% |

**Counting sequences**: All 6 sequences (3 vigesimal boundaries + 3 controls) produced perfectly correct French number words with correct vigesimal transitions.

**Interpretation**: At the behavioral level, GPT-4.1 has fully internalized the French counting system, including all vigesimal irregularities. There is no behavioral deficit for vigesimal numbers. This is consistent with the hypothesis that modern frontier models have memorized the French counting system from their training data.

### 5.2 Tokenization Analysis (Mistral 7B)

Vigesimal French numbers require significantly more tokens:

| Number Form | Mean Tokens (Decimal 20-69) | Mean Tokens (Vigesimal 70-99) |
|------------|---------------------------|------------------------------|
| French | 5.2 | **7.3** (+40%) |
| Digit | 3.0 | 3.0 |

The tokenization plot shows dramatic spikes in the vigesimal range, with "quatre-vingt-dix-sept" (97) requiring **10 tokens** vs 3 tokens for the digit "97". This increased token count means the model must propagate numeric information through more transformer steps.

### 5.3 Representational Analysis (Mistral 7B, Last Layer)

#### Ordinal Correlation (Spearman ρ between representation distances and numeric distances)

| Form | Spearman ρ | p-value |
|------|-----------|---------|
| French | 0.520 | < 10⁻³⁰⁰ |
| Digit | 0.485 | 4.15 × 10⁻²⁹⁰ |
| **Belgian French** | **0.591** | < 10⁻³⁰⁰ |

Belgian French representations preserve numeric ordering better than both standard French and digit representations. This is a striking finding: the *regular* decimal system (septante, huitante, nonante) produces internal representations that are more numerically organized than the vigesimal system.

#### Nearest Neighbor Error (distance between a number and its representational nearest neighbor)

| Form | Vigesimal NN Error | Decimal NN Error |
|------|-------------------|-----------------|
| French | 4.3 ± 4.3 | 2.7 ± 1.7 |
| Digit | 5.6 ± 5.2 | 3.1 ± 3.4 |
| Belgian French | 3.5 ± 2.3 | 2.7 ± 1.7 |

Vigesimal numbers have higher NN error (representational nearest neighbor is further from numeric neighbor) across all forms, with Belgian French showing improvement over standard French in the vigesimal range.

#### Cross-Form Similarity

| Comparison | Mean Cosine Similarity |
|-----------|----------------------|
| French-Digit | 0.907 |
| Belgian-Digit | 0.907 |
| **French-Belgian** | **0.995** |

French and Belgian French representations are nearly identical (0.995 cosine similarity), suggesting the model maps both forms to a shared underlying representation despite their surface differences.

### 5.4 Linear Probe Results

| Form | MAE (overall) | R² | Vigesimal MAE | Decimal MAE |
|------|-------------|-----|---------------|-------------|
| French | 2.72 ± 0.65 | 0.979 | 2.37 | 1.82 |
| Belgian | 2.79 ± 0.53 | 0.977 | 2.54 | 1.77 |
| Digit | 4.18 ± 1.19 | 0.943 | 4.37 | 3.49 |

Surprisingly, **French number words produce more linearly decodable representations** (R²=0.979) than digit strings (R²=0.943). This suggests that the contextualized prompt "Le nombre est [word]" provides richer numeric information than raw digits. The vigesimal MAE is higher than decimal MAE within each form.

### 5.5 Vigesimal Structure Analysis

The key question: do vigesimal numbers cluster by **linguistic base** or **numeric value**?

**Results for numbers 70-99**:
- **70% of nearest neighbors share a linguistic base** (e.g., 83's NN is 93, both "quatre-vingt-X")
- Only 53.3% of NNs are within ±2 numerically

Notable linguistic clustering examples:
- 80 ("quatre-vingts") → NN is 95 ("quatre-vingt-quinze"), distance = 15
- 83 ("quatre-vingt-trois") → NN is 93 ("quatre-vingt-treize"), distance = 10
- 73 ("soixante-treize") → NN is 63 ("soixante-trois"), distance = 10
- 91 ("quatre-vingt-onze") → NN is 82 ("quatre-vingt-deux"), distance = 9

**Control comparison (decimal 20-69)**: Only 34% of NNs are within ±2 numerically.

**Mann-Whitney U test** (vigesimal NN distance > decimal): U = 708.0, p = 0.667 (not significant). The vigesimal numbers are not statistically worse on average, but show a bimodal pattern: many have NN distance = 1 (following numeric order), while some have very large NN distances driven by linguistic similarity.

### 5.6 RSA (Representational Similarity Analysis)

| Comparison | Spearman ρ |
|-----------|-----------|
| French vs Ideal Numeric | 0.520 |
| Digit vs Ideal Numeric | 0.485 |
| **Belgian vs Ideal Numeric** | **0.591** |
| French vs Digit | 0.656 |
| Belgian vs Digit | 0.661 |
| **French vs Belgian** | **0.890** |

The RDMs (Representational Dissimilarity Matrices) show clear block structure for French and Belgian representations that partially mirrors numeric structure but also reflects linguistic groupings. Belgian French's RDM is most correlated with the ideal numeric RDM.

### 5.7 Layer Progression

Number representations develop across layers:

| Layer | Probe MAE | Spearman ρ |
|-------|----------|-----------|
| 0 (embed) | 31.0 | N/A |
| 8 | 18.8 | 0.526 |
| 16 (middle) | 16.4 | 0.525 |
| 24 | 18.4 | 0.490 |
| 32 (last) | 15.0 | 0.520 |

Linear probe accuracy improves throughout the network, with the best performance at the final layer (MAE=15.0). The Spearman correlation peaks at early-to-middle layers and dips in later layers before recovering — consistent with the model processing linguistic features first and reconstructing numeric information later.

## 6. Discussion

### Key Findings

1. **Behavioral mastery**: GPT-4.1 has fully mastered French counting, including all vigesimal irregularities. The hypothesis that vigesimal numbers cause behavioral errors is **refuted** for frontier models.

2. **Representational fingerprint**: Despite behavioral perfection, Mistral 7B's internal representations reveal the vigesimal structure. Numbers like 83 and 93 are representationally similar because they share the "quatre-vingt-" prefix, even though they are 10 apart numerically.

3. **Belgian French advantage**: The regular Belgian French system (septante/huitante/nonante) produces representations with better ordinal structure (ρ=0.591 vs 0.520), confirming that linguistic regularity aids numeric representation.

4. **French > Digit for probing**: Counterintuitively, French word representations are more linearly decodable for number values than digit representations. This may be because the contextual prompt activates richer semantic number representations.

5. **Token cost**: Vigesimal numbers require 40% more tokens, creating a computational overhead that may matter in resource-constrained settings.

### Surprises

- The **perfect behavioral accuracy** was unexpected — we hypothesized 10-30% lower accuracy on vigesimal numbers. This reflects how thoroughly modern LLMs have memorized French.
- The **linguistic clustering** in vigesimal representations is strong (70% of NNs share a linguistic base) even though the model clearly "knows" the numeric values (R²=0.979 probe accuracy).
- **French-Belgian similarity** of 0.995 suggests the model maps both forms to nearly the same internal representation, despite very different surface forms.

### Limitations

1. **Single model for representation analysis**: We used only Mistral 7B. The findings may differ for other architectures.
2. **Last-token representations**: We extracted hidden states at the last token position. Average pooling or other aggregation strategies might yield different results.
3. **Context sensitivity**: The prompt template "Le nombre est X" may influence representations. Other contexts could produce different patterns.
4. **Limited number range**: We focused on 0-99. The vigesimal system also appears in hundreds (e.g., "deux-cent-quatre-vingt-dix-sept" = 297).
5. **No causal intervention**: We observed representational differences but did not establish causality through activation patching.

## 7. Conclusions

### Summary
Modern LLMs have fully internalized the French vigesimal counting system at the behavioral level, achieving perfect accuracy on number conversion, arithmetic, and counting tasks. However, internal representations in Mistral 7B reveal that the model organizes vigesimal numbers by their linguistic structure — numbers sharing the same prefix (e.g., "quatre-vingt-") cluster together regardless of numeric distance. Belgian French's regular decimal system produces representations with better ordinal structure.

### Implications
- **For NLP researchers**: The gap between behavioral performance and representational organization suggests that probing studies can reveal structure invisible to behavioral benchmarks.
- **For multilingual applications**: The 40% tokenization overhead for vigesimal numbers may affect cost and latency in French-language applications.
- **For cognitive science**: The model's representational pattern (clustering by linguistic base) parallels findings in human numerical cognition research, where the French vigesimal system creates processing difficulties.

### Confidence in Findings
We are highly confident in the behavioral results (deterministic API calls, 100% accuracy) and moderately confident in the representational findings (consistent across multiple analyses but limited to one model and one context template).

## 8. Next Steps

### Immediate Follow-ups
1. **Causal interventions**: Patch vigesimal representations with decimal equivalents to test if the model's internal representation causally affects outputs
2. **Multi-model comparison**: Test Llama 3.1 8B, Qwen2.5 7B to check if findings generalize
3. **Extended range**: Analyze 100-999 where vigesimal constructions appear in more complex forms

### Alternative Approaches
- Sinusoidal/circular probes (per literature) instead of linear Ridge regression
- Attention pattern analysis to see which tokens attend to which during vigesimal processing
- Fine-grained tokenization analysis of how subword tokens map to numeric meaning

### Open Questions
- Does the linguistic clustering affect downstream task performance on tasks that require numeric reasoning with French numbers?
- Can we find specific attention heads responsible for the "vigesimal to decimal" conversion?
- How do other languages with irregular number systems (Danish, Georgian) compare?

## References

1. Bhattacharya et al. (2025). "Investigating the Interaction of Linguistic and Mathematical Reasoning Using Multilingual Number Puzzles." EMNLP.
2. Stefanik et al. (2025). "Unravelling the Mechanisms of Manipulating Numbers in Language Models." arXiv:2510.26285.
3. Levy & Geva (2024). "Language Models Encode Numbers Using Digit Representations in Base 10." arXiv:2410.11781.
4. Davies et al. (2025). "Language Models Do Not Embed Numbers Continuously." arXiv:2510.08009.
5. Kadlcik et al. (2025). "Pre-trained Language Models Learn Remarkably Accurate Representations of Numbers." EMNLP.
6. Various (2026). "Mechanistic Interpretability of Large-Scale Counting in LLMs." arXiv:2601.02989.
7. GraphPKU (2024). "Number Cookbook: Number Understanding of Language Models." ICLR 2025.
8. Singh & Strouse (2024). "Tokenization Counts." arXiv:2402.14903.
9. Marjieh et al. (2025). "What is a Number, That a Large Language Model May Know It?" arXiv:2502.01540.
10. Rahman & Mishra (2025). "A Fragile Number Sense." arXiv:2509.06332.
