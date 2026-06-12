# Sample domain patterns

These YAML files demonstrate the pattern-based Q&A generator used by
`prepare_training_data.py --yaml-dir sample_data/patternexamples`.

Each pattern describes a clinical topic with structured fields (description,
use cases, parameters, examples, common errors). The loader generates many
typed question-answer pairs automatically.

Add your own patterns following the same structure. See `data/medical_vocabulary.yaml`
for combinatorial vocabulary expansion at larger scale.
