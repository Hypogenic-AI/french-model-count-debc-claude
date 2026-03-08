# How Does the Model Count in French?

A research study investigating how LLMs represent and process the French vigesimal (base-20) counting system, where numbers 70-99 use irregular constructions like "quatre-vingt-dix-sept" (4x20+10+7 = 97).

## Key Findings

- **Perfect behavioral accuracy**: GPT-4.1 achieves 100% on all French number tasks (conversion, arithmetic, counting), including vigesimal numbers
- **Linguistic clustering in representations**: Mistral 7B internally organizes vigesimal numbers by linguistic prefix (70% of nearest neighbors share a base like "quatre-vingt-") rather than numeric proximity
- **Belgian French advantage**: Regular Belgian French (septante/huitante/nonante) produces representations with better ordinal structure (Spearman rho=0.591 vs 0.520 for standard French)
- **French words > digits for probing**: French number word representations are more linearly decodable (R2=0.979) than digit representations (R2=0.943)
- **Tokenization overhead**: Vigesimal numbers require 40% more tokens (7.3 vs 5.2 average)

## Project Structure

```
.
├── REPORT.md                 # Full research report with results
├── planning.md               # Research plan and methodology
├── literature_review.md      # Literature review (pre-gathered)
├── resources.md              # Resource catalog
├── src/
│   ├── experiment1_behavioral.py    # GPT-4.1 behavioral evaluation
│   ├── experiment2_representations.py # Mistral 7B hidden state analysis
│   ├── experiment3_analysis.py      # Statistical analysis & visualization
│   └── experiment4_probing.py       # Linear probing & RSA
├── results/
│   ├── behavioral_results.json      # Raw behavioral experiment results
│   ├── behavioral_stats.json        # Summary statistics
│   ├── representation_results.json  # Representation analysis results
│   ├── probing_results.json         # Probing & RSA results
│   └── plots/                       # All visualizations
├── datasets/french_numbers/         # Custom French number dataset
├── papers/                          # Downloaded research papers
└── code/                            # Cloned baseline repositories
```

## How to Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install openai numpy matplotlib scikit-learn scipy seaborn transformers torch accelerate

# Run experiments (requires OPENAI_API_KEY and GPU)
export USER=researcher  # needed for torch in some environments
python src/experiment1_behavioral.py    # ~5 min, requires OpenAI API
python src/experiment2_representations.py  # ~10 min, requires GPU
python src/experiment3_analysis.py      # ~1 min
python src/experiment4_probing.py       # ~10 min, requires GPU
```

## Models Used
- **GPT-4.1** (OpenAI API): Behavioral evaluation
- **Mistral 7B v0.3** (local, float16): Representational analysis

See [REPORT.md](REPORT.md) for full details.
