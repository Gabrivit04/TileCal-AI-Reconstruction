# TileCal AI Reconstruction Architecture

```text
RAW ROOT/ADC WINDOWS
        |
        v
 [Data Pipeline]
 pedestal subtraction -> normalization -> pile-up augmentation
        |
        v
 [Model Zoo]
 MLP | 1D-CNN (depthwise separable) | GRU
        |
        v
 [Dual-head Prediction]
 amplitude (A_pred), timing (tau_pred)
        |
        v
 [Hardware-aware Training]
 QAT / fake quantization (INT8/INT6/INT4 exploration)
        |
        v
 [hls4ml Export]
 HLS config generation -> C-sim/C-synth report ingestion
        |
        v
 [Pareto Metrics]
 physics resolution vs. latency vs. FPGA resources
```

## Repository principles
- **Physics-first metrics:** every model comparison includes amplitude and timing performance.
- **Firmware realism:** compact architectures and quantization are first-class citizens.
- **Reproducibility:** deterministic seeds, tests, and CI-driven checks.
