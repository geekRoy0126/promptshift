import inspect
import subprocess
import sys

import numpy as np

from promptshift import seeding
from promptshift.seeding import format_level, prompt_key, seed_from_key, stable_rng

ARGS = (42, "kvasir_seg", "cju0qkwl35piu0993l0dewei2", 0, "box_jitter", 0.10, 1)


def test_key_format_matches_spec():
    assert prompt_key(*ARGS) == "42|kvasir_seg|cju0qkwl35piu0993l0dewei2|0|box_jitter|0.1|1"


def test_level_formatting_is_canonical():
    assert format_level(0.10) == format_level(0.1) == format_level(1e-1) == "0.1"
    assert format_level(1) == format_level(1.0) == "1"
    assert format_level(1.25) == "1.25"
    assert format_level(None) == format_level(float("nan")) == "none"


def test_same_key_same_numbers_in_process():
    a = stable_rng(*ARGS).uniform(size=8)
    b = stable_rng(*ARGS).uniform(size=8)
    np.testing.assert_array_equal(a, b)


def test_different_keys_differ():
    base = stable_rng(*ARGS).uniform(size=4)
    for i, alt in [(4, "box_scale"), (5, 0.2), (6, 2), (3, 1), (2, "other_image")]:
        args = list(ARGS)
        args[i] = alt
        assert not np.array_equal(base, stable_rng(*args).uniform(size=4))


def test_same_key_same_seed_across_processes():
    # Python's hash() would fail this: it is salted per process.
    code = (
        "from promptshift.seeding import prompt_key, seed_from_key;"
        f"print(seed_from_key(prompt_key(*{ARGS!r})))"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert outputs == {str(seed_from_key(prompt_key(*ARGS)))}


def test_rng_does_not_depend_on_model():
    params = inspect.signature(stable_rng).parameters
    assert not any("model" in name for name in params)


def test_set_global_seed_is_repeatable():
    seeding.set_global_seed(7)
    a = np.random.rand(3)
    seeding.set_global_seed(7)
    np.testing.assert_array_equal(a, np.random.rand(3))
