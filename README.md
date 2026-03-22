# TileCal-AI-Reconstruction

AI-Accelerated Reconstruction for the ATLAS Tile Calorimeter at the HL-LHC.

## Why this repository is GSoC-ready
This project is designed as a **full technical showcase**: from physics-motivated preprocessing to compact deep learning models, then hardware-aware optimization and hls4ml-oriented export configuration.

### Core deliverables implemented in this bootstrap
1. **Modular data pipeline** for pulse preprocessing, normalization, augmentation, and pile-up emulation.
2. **Classical OF2 baseline** to compare ML against a physics-standard method.
3. **Model zoo** with compact MLP, depthwise-separable 1D-CNN, and GRU variants for 7/16 sample windows.
4. **QAT primitives** via fake quantization hooks for low-bit experiments.
5. **hls4ml export config generator** to standardize synthesis-ready metadata.
6. **Engineering quality**: tests, linting, CI workflow, docs, and a demo training script.

## Repository structure
```text
src/tilecal_ai/
  data/                # preprocessing and pile-up emulation
  baselines/           # OF2 implementation
  models/              # MLP, 1D-CNN, GRU
  training/            # custom multitask loss + trainer
  hardware/            # QAT helpers + hls4ml configuration

docs/
  ARCHITECTURE.md
  PROJECT_PLAN.md

tests/                 # pytest suite (with optional torch skips)
.github/workflows/     # CI: lint + tests
scripts/               # dummy training smoke-run
```

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

Optional ML extras:
```bash
pip install -e .[dev,ml]
python scripts/run_dummy_training.py
```

## Physics/firmware metrics to track next
- `sigma(ΔA / A_true)` vs pile-up `μ`
- timing residuals vs. phase shifts
- quantization degradation vs bit-width
- latency/resource Pareto frontier (DSP/LUT/FF/BRAM)

## Development standards
- PEP8-compatible style and `ruff` linting.
- Unit tests via `pytest`.
- CI run on every push and pull request.
- Notebook roadmap documented under `notebooks/README.md`.

---
If you are a GSoC mentor, this repo is intentionally structured to show both **scientific rigor** and **engineering maturity**, with a direct path to integration in an ATLAS-oriented production workflow.
