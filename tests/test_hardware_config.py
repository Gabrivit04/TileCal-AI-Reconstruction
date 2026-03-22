from __future__ import annotations

from tilecal_ai.hardware.hls4ml_export import HLSConfig, build_hls4ml_config


def test_hls4ml_config_contains_expected_keys() -> None:
    cfg = HLSConfig(clock_period_ns=4.0, precision="ap_fixed<6,2>")
    out = build_hls4ml_config("tilecal_cnn", input_len=7, cfg=cfg)

    assert out["Model"]["Name"] == "tilecal_cnn"
    assert out["Model"]["InputLength"] == 7
    assert out["Model"]["ClockPeriod"] == 4.0
    assert out["Model"]["Precision"] == "ap_fixed<6,2>"
    assert "dsp" in out["Metrics"]
