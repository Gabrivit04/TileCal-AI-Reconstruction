from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    sample_window: int = Field(default=7, description="ADC samples per channel window")
    channels: int = Field(default=1, description="Input channels per example")
    pileup_mu: float = Field(default=60.0, ge=0.0)


class ModelConfig(BaseModel):
    kind: str = Field(default="cnn1d", description="mlp | cnn1d | gru")
    hidden_dim: int = 32
    quant_bits: int = Field(default=8, ge=4, le=16)


class TrainingConfig(BaseModel):
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3


class ProjectConfig(BaseModel):
    experiment_name: str = "tilecal-hls-blueprint"
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    artifacts_dir: Path = Path("artifacts")
