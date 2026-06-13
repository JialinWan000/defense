# Repository Guidelines

## Project Structure & Module Organization

This repository contains PyTorch research code for JSCC/backdoor training and noise titration reproduction.

- `noise_titration/`: standalone reproduction package, including `Noise_Titration.py`, `README.md`, requirements, generated results, and plotting code.
- `noise_titration/result/Detection/`: saved JSON scores and `plot.py` for rendering detection plots.
- `models/`: shared model definitions for MAE, ViT-JSCC, and BDJSCC variants.
- `trainBackdoor/`: training entry points and epoch logic.
- `save/` and `ckpt/`: model weights and experiment checkpoints used by reproduction scripts.
- `paper/`: reference PDF material.

There is no dedicated test directory at present.

## Build, Test, and Development Commands

Create an environment and install runtime dependencies:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r noise_titration/requirements.txt
```

Regenerate the default noise titration JSON files:

```bash
python noise_titration/Noise_Titration.py
```

Run only the BDJSCC scan with a wider noise-power grid:

```bash
python noise_titration/Noise_Titration.py --model-kind bdjscc --r-start 0 --r-end 2 --r-step 0.1
```

Render a saved detection plot:

```bash
python noise_titration/result/Detection/plot.py noise_titration/result/Detection/bdjscc_noise_titration_scores.json
```

## Coding Style & Naming Conventions

Use Python with 4-space indentation. Prefer clear snake_case for functions, variables, and file names; keep model class names in CapWords when adding classes. Follow the existing script-oriented style and avoid broad refactors unless needed for the task. Keep paths configurable through arguments when practical, especially for datasets, checkpoints, and output files.

## Testing Guidelines

No formal test framework is configured. For changes to reproduction logic, run the relevant `Noise_Titration.py` command and verify that JSON output is produced under `noise_titration/result/Detection/`. For plotting changes, regenerate the PDF from a known JSON file. For training changes, use a short local run or dry-run path where possible before launching long GPU jobs.

## Commit & Pull Request Guidelines

This branch has no commit history yet, so there is no established commit convention. Use short, imperative commit messages such as `Add BDJSCC plot regeneration docs` or `Fix checkpoint path handling`. Pull requests should describe the experiment or code path affected, list commands run, note required checkpoints or datasets, and include before/after plots when visual outputs change.

## Security & Configuration Tips

Do not commit private datasets, new large checkpoints, credentials, or machine-specific absolute paths. Prefer documenting required external files and command-line arguments over hard-coding local paths.
