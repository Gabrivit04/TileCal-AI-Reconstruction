# TileCal AI Reconstruction Architecture

```mermaid
flowchart LR
  A[ROOT ADC samples] --> B[Data Pipeline<br/>uproot + awkward + torch Dataset]
  B --> C[Models: MLP / 1D-CNN / Tiny GRU]
  C --> D[Joint Loss<br/>Amplitude + Timing]
  D --> E[QAT INT8/INT6/INT4]
  E --> F[hls4ml conversion]
  F --> G[HLS reports: latency, LUT, DSP, BRAM]
```

## Performance Axes
- Physics quality: `sigma((A_pred - A_true)/A_true)` vs pile-up.
- Latency target: below `100 ns / channel` where feasible.
- FPGA resources: LUT, FF, DSP48, BRAM utilization.
