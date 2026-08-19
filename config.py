from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Paths:
    raw: Path = PROJECT_ROOT / "data" / "raw"
    processed: Path = PROJECT_ROOT / "data" / "processed"
    manifests: Path = PROJECT_ROOT / "out" / "manifests"
    checkpoints: Path = PROJECT_ROOT / "out" / "checkpoints"
    eval: Path = PROJECT_ROOT / "out" / "eval"
    bench: Path = PROJECT_ROOT / "out" / "bench"
    models: Path = PROJECT_ROOT / "out" / "models"

    def make(self) -> None:
        for path in self.__dict__.values():
            path.mkdir(parents=True, exist_ok=True)


PATHS = Paths()

DEFAULT_CMB_ZIP_URL = (
    "https://huggingface.co/datasets/FreedomIntelligence/CMB/resolve/main/"
    "CMB-datasets.zip?download=true"
)
DEFAULT_MODEL_ID = os.environ.get("PROJ3_MODEL_ID", "Qwen/Qwen3.5-9B-Base")
DEFAULT_MAX_LENGTH = int(os.environ.get("PROJ3_MAX_LENGTH", "4096"))
DEFAULT_SEED = int(os.environ.get("PROJ3_SEED", "20260818"))

