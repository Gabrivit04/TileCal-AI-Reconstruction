# TileCal-AI-Reconstruction

> **AI-Accelerated Reconstruction for the ATLAS Tile Calorimeter at the HL-LHC**  
> A GSoC-ready blueprint from ROOT ingestion to FPGA-aware ML deployment.

## Why this repository stands out
This project is designed as an **end-to-end, reproducible research and engineering pipeline** aligned with ATLAS TileCal trigger constraints:
- Modular data processing from pulse samples to tensors.
- Classical baseline (Optimal Filtering 2) for fair physics comparison.
- Compact neural architectures tailored to 7/16-sample windows.
- Quantization-aware path toward low-bit hardware implementation.
- hls4ml export hooks for synthesis and latency/resource studies.
- CI + tests + clear milestones to guarantee maintainability.

## Project objectives and deliverables
1. **Modular Data Pipeline** (`tilecal_ai.data`): ingestion-ready preprocessing, pedestal subtraction, normalization, and runtime augmentation.
2. **Baseline Implementation** (`tilecal_ai.baselines`): OF2 model for amplitude and timing reconstruction.
3. **Advanced ML Models** (`tilecal_ai.models`): MLP, depthwise-separable 1D-CNN, and tiny GRU.
4. **Hardware-Aware Optimization** (`tilecal_ai.quantization`): QAT preparation and INT8 conversion hooks.
5. **FPGA Synthesis via hls4ml** (`tilecal_ai.hls`): hls4ml config export and integration points.
6. **Comprehensive Documentation** (`docs/`, `notebooks/`): architecture, notebooks roadmap, and reproducible workflow.

## Repository structure

```text
.
├── src/tilecal_ai/
│   ├── data/pipeline.py             # dataset + preprocessing + augmentation
│   ├── baselines/optimal_filtering.py
│   ├── models/networks.py
│   ├── training/losses.py
│   ├── training/train.py            # smoke training loop
│   ├── quantization/qat.py
│   └── hls/export.py
├── tests/test_pipeline_and_models.py
├── docs/architecture.md
├── notebooks/README.md
└── .github/workflows/ci.yml
```

## Technical methodology
- **Preprocessing**: pedestal subtraction + event-wise normalization.
- **Augmentation**: configurable Gaussian noise and phase shift emulation.
- **Learning target**: joint regression for amplitude (`A_pred`) and timing (`tau_pred`).
- **Loss**: multi-task weighted Huber/MSE loss.
- **Model candidates**:
  - MLP (strong speed baseline),
  - depthwise-separable 1D-CNN (primary hardware-friendly candidate),
  - tiny GRU (accuracy-oriented with latency trade-off).

See architecture diagram in `docs/architecture.md`.

## 350-hour GSoC timeline (sprint-oriented)
- **Phase 0 (Community Bonding)**: environment, ATLAS docs, data schema, docker reproducibility.
- **Phase 1 (Weeks 1-4)**: data pipeline + OF2 baseline + benchmark plots.
- **Phase 2 (Weeks 5-8)**: FP32 model prototyping + hyperparameter sweeps.
- **Phase 3 (Weeks 9-14)**: QAT + hls4ml + synthesis profiling.
- **Phase 4 (Weeks 15-16+)**: hardening, tests, documentation, handoff artifacts.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
python -m tilecal_ai.training.train
```

## Reproducibility and CI/CD
- **Unit tests** with `pytest` on data, models, loss, OF2, and HLS config export.
- **CI pipeline** runs lint + tests + 1-epoch training smoke check.
- Supports experiment tracking extension via MLflow (`pip install -e .[track]`).

## Next technical increments
1. Plug real ROOT readers (uproot/awkward) with configurable schema maps.
2. Implement full OF2 coefficient derivation from pulse templates and noise autocorrelation.
3. Add Optuna sweep scripts for model/hyperparameter search.
4. Export trained models to ONNX and integrate full hls4ml conversion scripts.
5. Add report-ready benchmarking notebooks: physics resolution, latency, and resources.

---
If you'd like, I can generate the **first production notebook set**, a **Docker-based dev environment**, and a **mentors-facing technical report template** in the next iteration.
