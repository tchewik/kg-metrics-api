#!/usr/bin/env python3
"""Fail fast when a supposedly self-contained image is missing model artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer


def fail(message: str) -> None:
    print(f"[metrics-api] ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def verify_file(path: Path, *, min_bytes: int = 1) -> None:
    if not path.is_file():
        fail(f"required model file is missing: {path}")
    size = path.stat().st_size
    if size < min_bytes:
        fail(f"required model file looks incomplete: {path} ({size} bytes)")


def verify_dir(path: Path, required_children: tuple[str, ...]) -> None:
    if not path.is_dir():
        fail(f"required model directory is missing: {path}")
    missing = [name for name in required_children if not (path / name).exists()]
    if missing:
        fail(f"{path} is incomplete; missing: {', '.join(missing)}")


def verify_hf_repo(repo_id: str, cache_dir: Path) -> None:
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir),
                local_files_only=True,
            )
        )
        AutoConfig.from_pretrained(
            repo_id,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
        AutoTokenizer.from_pretrained(
            repo_id,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
    except Exception as exc:  # dependency exception classes vary by version
        fail(
            f"Hugging Face model {repo_id!r} is not usable from local cache "
            f"{cache_dir}: {exc}"
        )

    weight_candidates = (
        list(snapshot.glob("*.safetensors"))
        + list(snapshot.glob("pytorch_model*.bin"))
        + list(snapshot.glob("*.safetensors.index.json"))
        + list(snapshot.glob("pytorch_model.bin.index.json"))
    )
    if not weight_candidates:
        fail(f"Hugging Face model {repo_id!r} has no cached model weights in {snapshot}")

    print(f"[metrics-api] bundled HF model OK: {repo_id} -> {snapshot}")


def main() -> None:
    align = Path(os.environ.get("ALIGN_CKPT_PATH", "/models/AlignScore-large.ckpt"))
    bleurt = Path(os.environ.get("BLEURT_CHECKPOINT", "/models/BLEURT-20"))
    hf_cache = Path(os.environ.get("TRANSFORMERS_CACHE", "/cache/huggingface"))

    # A real AlignScore-large checkpoint is multi-GB. The 1 MiB floor catches
    # missing/empty files and accidental Git-LFS pointer files without pinning size.
    verify_file(align, min_bytes=1024 * 1024)

    verify_dir(bleurt, ("bleurt_config.json", "saved_model.pb", "variables"))
    verify_file(bleurt / "variables" / "variables.index")
    if not any((bleurt / "variables").glob("variables.data-*")):
        fail(f"{bleurt / 'variables'} has no variables.data-* checkpoint file")

    configured = os.environ.get(
        "BUNDLED_HF_MODELS",
        "roberta-large;Inria-CEDAR/FactSpotter-DeBERTaV3-Base",
    )
    for repo_id in (part.strip() for part in configured.split(";")):
        if repo_id:
            verify_hf_repo(repo_id, hf_cache)

    print("[metrics-api] bundled model verification passed")


if __name__ == "__main__":
    main()
