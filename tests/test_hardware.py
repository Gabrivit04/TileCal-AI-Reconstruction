from tilecal_ai.hardware.quantization import recommended_precision


def test_recommended_precision_balanced():
    spec = recommended_precision("balanced")
    assert spec.total_bits == 8
    assert "ap_fixed" in spec.hls_type
