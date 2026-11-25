"""
xmad/4_iteration/xmad_bench -- benchmark adapters (C3.3).

    from xmad_bench import load_benchmark
    examples = load_benchmark("musique", "answerable", "dev")   # -> list[Example]

Each adapter turns its benchmark into Example objects (base.py). Adding a
benchmark = one module with load(config, split, data_dir) + one entry in
ADAPTERS and in config/base.yaml.
"""

from pathlib import Path

from . import hotpotqa, musique, twowiki
from .base import Example
from .scoring import em_max, evidence_recall, f1_max

ADAPTERS = {"hotpotqa": hotpotqa, "musique": musique, "2wiki": twowiki}
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_benchmark(name, config, split, data_dir=DATA_DIR):
    if name not in ADAPTERS:
        raise KeyError(f"unknown benchmark {name!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[name].load(config=config, split=split, data_dir=data_dir)


__all__ = ["Example", "load_benchmark", "em_max", "f1_max", "evidence_recall", "ADAPTERS"]
