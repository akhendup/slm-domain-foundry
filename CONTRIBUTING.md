# Contributing to SLM Domain Foundry

Thank you for helping improve this project. This repository is intended as a
clean, domain-adaptive starter kit for teams building small language models.

## Getting started

1. Fork the repository and create a feature branch.
2. Create a virtual environment (Python 3.10+):

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Run the test suite before opening a pull request:

   ```bash
   pytest tests/ --tb=short
   ```

## Development guidelines

- Keep changes focused and domain-agnostic unless they belong in sample data or examples.
- Put domain-specific keywords, prompts, and extraction patterns in `domain_config.yaml`
  (or a new file under `examples/`), not hardcoded in Python.
- Update `config.yaml`, README, and tests when changing CLI behavior or defaults.
- Add tests for new behavior. Target ≥75% coverage on changed modules.

## Reporting issues

Open a GitHub issue with:

- A clear description of the problem or feature request
- Steps to reproduce (for bugs)
- Python version, OS, and relevant config snippets

## Pull requests

- Link related issues when applicable.
- Describe what changed and why.
- Confirm tests pass locally.
- For Phase 2 enhancements (synthetic data, ORPO, DAPT, etc.), include benchmark notes
  or validation steps in the PR description.

## Code of conduct

Be respectful and constructive. Medical AI contributions should prioritize safety,
accuracy, and clear limitations in documentation and sample prompts.
