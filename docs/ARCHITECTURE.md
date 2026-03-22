# Architecture Blueprint

```text
Raw ADC samples -> preprocessing -> baseline OF2 + ML inference -> QAT -> hls4ml export -> FPGA metrics
```

## Core tracks
- **Data pipeline**: ROOT ingestion (planned via uproot), pedestal subtraction, normalization, and pile-up aware augmentation.
- **Algorithm baselines**: OF2 reference implementation to compare amplitude and timing reconstruction.
- **ML models**: compact MLP / CNN1D / GRU candidates designed for 7- and 16-sample windows.
- **Hardware path**: quantization profiles + hls4ml handoff targeting low latency and low DSP usage.

## Deliverable mapping
1. Data ingestion and cleaning module.
2. OF2 baseline and benchmark scripts.
3. PyTorch model zoo for compact architectures.
4. Quantization policy layer and evaluation notebooks.
5. hls4ml conversion entrypoints and synthesis report parser (planned).
6. Reproducibility with tests, CI, and notebooks.
