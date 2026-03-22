# GSoC Execution Plan (350h)

## Phase 0 — Community Bonding (May 8 - June 1)
- Setup Dockerized development environment.
- Study TileCal Phase-II constraints and existing IFIC/hls4ml workflows.
- Freeze data loader interfaces.

## Phase 1 — Data + Baselines (June 2 - June 29)
- Build vectorized ingestion and preprocessing interfaces.
- Implement and benchmark OF2 baseline.
- Produce initial resolution plots vs. energy and pile-up.

## Phase 2 — ML Development (June 30 - July 27)
- Implement MLP/CNN/GRU compact candidates.
- Train with multi-task objective (amplitude + timing).
- Hyperparameter sweeps and midterm model freeze.

## Phase 3 — Hardware-aware Optimization (July 28 - Sept 7)
- Quantization-aware training down to low precision.
- hls4ml conversion and C simulation parity checks.
- Synthesis profiling: latency, DSP/LUT/FF/BRAM.

## Phase 4 — Finalization (Sept 8 - Oct)
- Strengthen tests + CI and polish docs.
- Final report, reproducible notebooks, and handoff.
