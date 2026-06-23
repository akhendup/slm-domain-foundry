# Contributing to SLM Domain Foundry

Thank you for helping improve this project. This repository is intended as a
clean, domain-adaptive starter kit for teams building small language models.
You do **not** need any commercial chat-API subscription to develop or run the pipeline.

## Getting started

1. Fork the repository and create a feature branch.
2. Create a virtual environment (Python 3.10+):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

   **Apple Silicon (MPS):** use the native stack — **not Docker** (containers cannot access Metal/MPS on Mac). See README [Hardware & platforms](../README.md#hardware--platforms).

   ```bash
   pip install -r requirements-mps.txt
   pip install -e ".[dev]"
   # or: pip install -e ".[mps,dev]"
   ./run_local.sh
   ```

3. Run the test suite before opening a pull request:

   ```bash
   pytest tests/ --tb=short
   ./scripts/security_scan.sh   # recommended before release-related PRs
   ```

## Development guidelines

- Keep changes focused and domain-agnostic unless they belong in sample data or examples.
- Put domain-specific keywords, prompts, and extraction patterns in `domain_config.yaml`
  (or a new file under `examples/`), not hardcoded in Python.
- Update `config.yaml`, README, and tests when changing CLI behavior or defaults.
- Add tests for new behavior. Target ≥75% coverage on changed modules.

## Reporting issues

Open an issue on [GitHub Issues](https://github.com/akhendup/slm-domain-foundry/issues) with:

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
