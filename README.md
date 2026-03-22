# TileCal-AI-Reconstruction

> A **mentor-facing, production-minded blueprint repository** for GSoC: from TileCal pulse samples to hardware-aware ML inference and FPGA synthesis constraints.

## Why this repo stands out

This repository is intentionally structured to demonstrate not only model experimentation but also **scientific reproducibility**, **firmware awareness**, and **engineering discipline** expected in ATLAS-scale workflows.

### What is already implemented
- ✅ Modular Python package (`src/tilecal_ai`) with clear domain boundaries.
- ✅ Data preprocessing primitives for pedestal subtraction + normalization.
- ✅ Classical Optimal Filtering baseline (`OF2`-style linear estimator).
- ✅ Compact ML model templates (MLP and CNN1D).
- ✅ Multi-task loss for amplitude + timing prediction.
- ✅ Hardware precision policy helper producing HLS fixed-point types.
- ✅ Unit-test suite + CI pipeline with lint, tests, and smoke train step.

### Planned extensions (aligned with TileCal Phase-II goals)
- ROOT ingestion using `uproot` + `awkward`.
- Pile-up aware dynamic augmentation in dataloaders.
- GRU comparison for latency/accuracy trade-offs.
- QAT and ONNX/hls4ml conversion scripts.
- Vivado report parser and Pareto plots for latency/resources/performance.

---

## Architecture snapshot

```text
Raw ADC samples
   -> preprocessing (pedestal subtraction, normalization, augmentation)
   -> [OF2 baseline] + [Tiny MLP/CNN/GRU candidates]
   -> multi-task outputs (Amplitude, Time)
   -> quantization policies (INT16..INT4)
   -> hls4ml export
   -> synthesis metrics (latency, DSP, LUT, FF)
```

See also: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Quickstart

```bash
python -m pip install -e .[dev]
pytest
python scripts/bootstrap_demo.py
```

---

## Repository layout

```text
.github/workflows/ci.yml   # CI: lint + tests + smoke loop
src/tilecal_ai/
  baseline/                # OF2 baseline
  data/                    # preprocessing + datasets
  models/                  # compact neural networks
  training/                # losses / training utilities
  hardware/                # fixed-point and hardware-aware helpers
tests/                     # unit tests
scripts/                   # executable demos and utilities
docs/                      # architecture and roadmap
notebooks/                 # reproducible analysis plan
```

---

## Scientific engineering principles

- **Reproducibility-first:** deterministic seeds and traceable artifacts.
- **Benchmark honesty:** OF2 baseline kept as first-class citizen.
- **Hardware realism:** every model decision is tied to latency/resource implications.
- **Mentor readability:** modular design, explicit roadmap, and CI-enforced quality gates.

---

## 16-week execution roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the sprint-level plan aligned with your project statement (data → baseline → ML → QAT → hls4ml → docs).

---

## License

MIT
