# Downloaded Datasets

This directory contains datasets for the research project on how LLMs count in French.
Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: French Numbers (0-999)

### Overview
- **Source**: Custom-generated for this research project
- **Size**: 1,000 number entries + 1,000 arithmetic tasks + 48 counting prompts + 499 comparison tasks
- **Format**: JSON
- **Task**: Number representation probing, arithmetic evaluation, counting evaluation
- **License**: Research use

### Files
- `french_numbers/french_numbers_0_999.json` — Complete mapping of numbers 0-999 to French words, Belgian French words, with category labels and implicit operation annotations
- `french_numbers/vigesimal_subset_70_99.json` — Focused subset of vigesimal-range numbers (70-99)
- `french_numbers/arithmetic_tasks_sample.json` — 1,000 addition tasks in French (a + b = ?)
- `french_numbers/counting_prompts.json` — 48 counting sequence tasks spanning different ranges
- `french_numbers/comparison_tasks.json` — 499 number comparison tasks in French
- `french_numbers/samples.json` — Representative sample entries

### Schema

Each entry in `french_numbers_0_999.json`:
```json
{
  "number": 97,
  "french_word": "quatre-vingt-dix-sept",
  "belgian_french_word": "nonante-sept",
  "english_digit": "97",
  "category": "vigesimal_90s",
  "uses_vigesimal": true,
  "implicit_operations": ["4 × 20 + 17", "multiplication", "addition"]
}
```

### Loading
```python
import json
with open("datasets/french_numbers/french_numbers_0_999.json", "r") as f:
    data = json.load(f)

# Filter vigesimal numbers
vigesimal = [e for e in data if e["uses_vigesimal"]]
```

### Categories
- `unique` (0-16): Unique French words
- `teens` (17-19): dix-sept, dix-huit, dix-neuf
- `decimal_tens` (20-69): Standard decimal system
- `vigesimal_70s` (70-79): soixante-dix system (60 + 10-19)
- `vigesimal_80s` (80-89): quatre-vingts system (4 × 20 + 0-9)
- `vigesimal_90s` (90-99): quatre-vingt-dix system (4 × 20 + 10-19)
- `hundreds` (100-999): Standard hundreds

### Design Notes
- Belgian/Swiss French variants (septante, huitante, nonante) provide natural controls for studying the effect of vigesimal vs. decimal structure
- Implicit operations are annotated for each vigesimal number to support analysis of compositional processing
- Arithmetic tasks include flags for whether inputs/outputs involve vigesimal numbers
