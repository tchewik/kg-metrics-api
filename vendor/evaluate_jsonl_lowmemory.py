#!/usr/bin/env python
"""Low-memory wrapper around evaluate_jsonl.py.
Runs each heavy metric in its own subprocess so GPU memory is released between metrics.
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from evaluate_jsonl import fmt_compact
except ModuleNotFoundError:
    def fmt_compact(x: float, sigfigs: int = 4) -> str:
        import math
        if not math.isfinite(x):
            return str(x)
        ax = abs(x)
        if ax != 0 and (ax < 1e-3 or ax >= 1e4):
            return f"{x:.{sigfigs}e}"
        return f"{x:.{sigfigs}f}"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [lowmemory wrapper]: %(message)s",
)


def filter_forward_args(args: List[str]) -> List[str]:
    filtered: List[str] = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a == "--output-path":
            skip_next = True
            continue
        if a.startswith("--no-"):
            continue
        filtered.append(a)
    return filtered


def run_single_metric(
    metric_label: str,
    input_path: Path,
    evaluator_script: Path,
    base_args: List[str],
    metric_specific_flags: List[str],
    tmpdir: Path,
) -> Dict[str, Any]:
    out_path = tmpdir / f"{metric_label}.json"
    cmd = [
        sys.executable,
        str(evaluator_script),
        str(input_path),
        "--output-path",
        str(out_path),
        *base_args,
        *metric_specific_flags,
    ]
    logger.info("Running %s with command:\n  %s", metric_label, " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not out_path.exists():
        logger.warning("Expected output %s not found for %s.", out_path, metric_label)
        return {}

    with out_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-memory wrapper for evaluate_jsonl.py")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--evaluator-script",
        type=Path,
        default=Path(__file__).with_name("evaluate_jsonl.py"),
    )
    parser.add_argument("--no-bleu", action="store_true")
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--no-bleurt", action="store_true")
    parser.add_argument("--no-infolm", action="store_true")
    parser.add_argument("--no-alignscore", action="store_true")
    parser.add_argument("--no-factspotter", action="store_true")
    args, unknown = parser.parse_known_args()

    if not args.evaluator_script.exists():
        parser.error(f"Evaluator script not found: {args.evaluator_script}")

    base_args = filter_forward_args(unknown)
    metric_runs: Dict[str, List[str]] = {}

    if not args.no_bleu:
        metric_runs["BLEU"] = ["--no-bertscore", "--no-alignscore", "--no-factspotter", "--no-bleurt", "--no-infolm"]
    if not args.no_bertscore:
        metric_runs["BERTScore"] = ["--no-bleu", "--no-alignscore", "--no-factspotter", "--no-bleurt", "--no-infolm"]
    if not args.no_bleurt:
        metric_runs["BLEURT"] = ["--no-bleu", "--no-bertscore", "--no-alignscore", "--no-factspotter", "--no-infolm"]
    if not args.no_infolm:
        metric_runs["InfoLM"] = ["--no-bleu", "--no-bertscore", "--no-alignscore", "--no-factspotter", "--no-bleurt"]
    if not args.no_alignscore:
        metric_runs["AlignScore"] = ["--no-bleu", "--no-bertscore", "--no-factspotter", "--no-bleurt", "--no-infolm"]
    if not args.no_factspotter:
        metric_runs["FactSpotter"] = ["--no-bleu", "--no-bertscore", "--no-alignscore", "--no-bleurt", "--no-infolm"]

    if not metric_runs:
        parser.error("All metrics are disabled; nothing to run.")

    combined: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for metric_label, flags in metric_runs.items():
            data = run_single_metric(
                metric_label=metric_label,
                input_path=args.input_path,
                evaluator_script=args.evaluator_script,
                base_args=base_args,
                metric_specific_flags=flags,
                tmpdir=tmpdir,
            )
            combined.update(data)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    logger.info("Wrote merged metrics to %s", args.output_path)
    logger.info("Overall metrics:")
    for k, v in combined.items():
        if k == "BLEU":
            logger.info("  %s = %s", k, fmt_compact(v["micro-avg"]))
        elif k == "BERTScore":
            logger.info("  %s = %s", k, fmt_compact(v["f1"]["mean"] * 100))
        elif k in {"BLEURT", "InfoLM"}:
            logger.info("  %s = %s", k, fmt_compact(v["mean"]))
        elif k in {"AlignScore", "FactSpotter"}:
            logger.info("  %s = %s", k, fmt_compact(v["mean"] * 100))
        else:
            logger.info("  %s = %.2f", k, np.round(v, 2))


if __name__ == "__main__":
    main()
