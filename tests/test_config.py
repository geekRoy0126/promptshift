import math

import yaml

from promptshift.paths import REPO_ROOT


def test_default_config_loads_and_matches_spec():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    assert cfg["global_seed"] == 42
    assert cfg["data"]["mask_threshold"] == 127
    assert cfg["data"]["min_object_frac"] == 0.001
    assert cfg["metrics"]["nsd_tolerance_px"] == 2
    jitter = cfg["prompts"]["box_jitter"]
    scale = cfg["prompts"]["box_scale"]
    assert jitter["levels"] == [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
    assert jitter["seeds"] == [0, 1, 2]
    assert scale["levels"] == [0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
    # 6 x 3 jitter + 6 scale = 24 configurations per image
    assert len(jitter["levels"]) * len(jitter["seeds"]) + len(scale["levels"]) * len(
        scale["seeds"]
    ) == 24
    assert math.isclose(sum(cfg["data"]["split"]["fractions"].values()), 1.0)
