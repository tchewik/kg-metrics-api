#!/usr/bin/env python
"""Small client for sending a predictions JSON/JSONL file to the remote metrics container."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input_path", type=Path)
    p.add_argument("--api-url", type=str, default="http://localhost:8125/v1/evaluate/file")
    p.add_argument("--output-path", type=Path, required=True)
    p.add_argument("--metrics", type=str, default="bleu,bertscore,bleurt,alignscore,factspotter")
    p.add_argument("--bertscore-lang", type=str, default="en")
    p.add_argument("--align-ckpt-path", type=str, default="/models/AlignScore-large.ckpt")
    p.add_argument("--bleurt-checkpoint", type=str, default="/models/BLEURT-20")
    p.add_argument("--align-model", type=str, default="roberta-large")
    p.add_argument("--align-batch-size", type=int, default=8)
    p.add_argument("--align-eval-mode", type=str, default="nli_sp")
    p.add_argument("--factspotter-model-name", type=str, default="Inria-CEDAR/FactSpotter-DeBERTaV3-Base")
    p.add_argument("--factspotter-batch-size", type=int, default=16)
    p.add_argument("--factspotter-entailment-threshold", type=float, default=0.5)
    p.add_argument("--infolm-model-name", type=str, default="bert-base-uncased")
    p.add_argument("--infolm-information-measure", type=str, default="kl_divergence")
    p.add_argument("--infolm-temperature", type=float, default=0.25)
    p.add_argument("--infolm-batch-size", type=int, default=64)
    p.add_argument("--timeout-sec", type=int, default=7200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = {
        "metrics": args.metrics,
        "bertscore_lang": args.bertscore_lang,
        "align_ckpt_path": args.align_ckpt_path,
        "bleurt_checkpoint": args.bleurt_checkpoint,
        "align_model": args.align_model,
        "align_batch_size": str(args.align_batch_size),
        "align_eval_mode": args.align_eval_mode,
        "factspotter_model_name": args.factspotter_model_name,
        "factspotter_batch_size": str(args.factspotter_batch_size),
        "factspotter_entailment_threshold": str(args.factspotter_entailment_threshold),
        "infolm_model_name": args.infolm_model_name,
        "infolm_information_measure": args.infolm_information_measure,
        "infolm_temperature": str(args.infolm_temperature),
        "infolm_batch_size": str(args.infolm_batch_size),
    }

    with args.input_path.open("rb") as f:
        resp = requests.post(
            args.api_url,
            data=data,
            files={"file": (args.input_path.name, f, "application/octet-stream")},
            timeout=args.timeout_sec,
        )
    resp.raise_for_status()
    payload = resp.json()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload["metrics"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"request_id": payload["request_id"], "duration_sec": payload["duration_sec"]}, indent=2))


if __name__ == "__main__":
    main()
