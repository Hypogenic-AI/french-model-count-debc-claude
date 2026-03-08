# Research Plan: How Does the Model Count in French?

## Motivation & Novelty Assessment

### Why This Research Matters
French has a famously irregular counting system where numbers 70-99 use vigesimal (base-20) constructions: "soixante-dix" (60+10=70), "quatre-vingts" (4×20=80), "quatre-vingt-dix" (4×20+10=90). This creates a natural experiment: do LLMs represent these numbers by their linguistic surface form or by their underlying numeric value? Understanding this reveals how LLMs bridge symbolic language and mathematical reasoning.

### Gap in Existing Work
Per the literature review, no study has specifically probed French number representations in LLMs. Prior work shows: (1) models use base-10 circular representations internally (Levy & Geva 2024), (2) implicit compositional operators are a bottleneck (Bhattacharya et al. 2025), (3) sinusoidal probes can recover number values from hidden states (Stefanik et al. 2025). But nobody has tested whether vigesimal French numbers create representational difficulties.

### Our Novel Contribution
We conduct the first systematic study of how LLMs handle French vigesimal counting through two complementary approaches:
1. **Behavioral evaluation**: Testing GPT-4.1 on French number tasks to measure accuracy differences between vigesimal (70-99) and decimal (20-69) ranges
2. **Representational analysis**: Using a local model (Mistral 7B) to extract and visualize internal representations of French numbers, comparing vigesimal vs. decimal vs. Belgian French vs. digit forms

### Experiment Justification
- **Experiment 1 (Number Conversion)**: Tests whether models can correctly map French words to numbers and back, isolating the vigesimal difficulty
- **Experiment 2 (Arithmetic in French)**: Tests whether vigesimal numbers cause more arithmetic errors
- **Experiment 3 (Counting Sequences)**: Tests whether models can generate correct sequences across vigesimal boundaries (69→70, 79→80, 89→90)
- **Experiment 4 (Representation Analysis)**: Maps internal representations to understand *why* errors occur — is the model confused about the value or just the linguistic form?

## Research Question
How do LLMs internally represent and process French vigesimal numbers (70-99), and does the irregular counting system cause systematic difficulties compared to regular decimal numbers?

## Hypothesis Decomposition
H1: Models make more errors on vigesimal French numbers (70-99) than on decimal numbers (20-69) in conversion tasks
H2: Arithmetic involving vigesimal numbers produces more errors than purely decimal arithmetic
H3: Counting sequences break at vigesimal boundaries (69→70, 79→80, 89→90)
H4: Internal representations of vigesimal French numbers are more distant from their true numeric value than decimal numbers
H5: Belgian French equivalents (septante, huitante, nonante) produce representations closer to true numeric value than standard French vigesimal forms

## Proposed Methodology

### Approach
Two-pronged: behavioral evaluation via API + representational analysis via local model.

### Experimental Steps
1. **Setup**: Install dependencies, prepare datasets
2. **Exp 1 - Conversion**: GPT-4.1 converts French→digit and digit→French for 0-99, measuring accuracy by vigesimal status
3. **Exp 2 - Arithmetic**: GPT-4.1 performs addition in French, comparing vigesimal-involving vs decimal-only tasks
4. **Exp 3 - Counting**: GPT-4.1 generates counting sequences, testing boundary crossings
5. **Exp 4 - Representations**: Extract Mistral 7B hidden states for French/Belgian/English/digit number forms, visualize with t-SNE/PCA, measure representational distances

### Baselines
- Decimal range (20-69) as control for vigesimal range (70-99)
- Belgian French as control for standard French
- English words and digit strings as additional reference points

### Evaluation Metrics
- Exact match accuracy for conversion/arithmetic/counting tasks
- Error rate by number category (vigesimal vs decimal)
- Cosine similarity between number representations and ground truth ordering
- t-SNE/PCA visualization of representation space

### Statistical Analysis Plan
- Chi-squared tests for accuracy differences between vigesimal and decimal groups
- Effect sizes (Cohen's h for proportions)
- Bootstrap confidence intervals for accuracy estimates

## Expected Outcomes
- Vigesimal numbers will show 10-30% lower accuracy on behavioral tasks
- Belgian French will perform closer to decimal French
- Internal representations will show the vigesimal numbers are noisier/more distant from ordinal ground truth

## Timeline and Milestones
1. Environment setup + data prep: 10 min
2. Behavioral experiments (API calls): 30 min
3. Representational analysis (local model): 45 min
4. Analysis + visualization: 30 min
5. Documentation: 20 min

## Potential Challenges
- API rate limits → batch with delays
- Model download time for Mistral 7B → use smaller model if needed
- Token limits for long French number words → test beforehand

## Success Criteria
- Clear evidence of differential performance on vigesimal vs decimal numbers
- Interpretable visualization of the representational structure
- Statistical tests with p < 0.05 for key comparisons
