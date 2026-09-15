#!/usr/bin/env python
"""
Evaluate KG→text predictions stored in JSON / JSONL.
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import os
os.environ["PYTORCH_JIT"] = "0"  # FactSpotter fix for some Python/runtime combos

import argparse
import gc
import json
import logging
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(module)s.py: %(message)s",
)


import nltk


def require_nltk_resource(resource_path: str, install_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError as exc:
        raise RuntimeError(
            f"Missing NLTK resource {install_name!r}. "
            f"Install it at build time, for example:\n"
            f"python -m nltk.downloader -d /cache/nltk_data {install_name}\n"
            f"and run with NLTK_DATA=/cache/nltk_data"
        ) from exc


require_nltk_resource("tokenizers/punkt_tab/english/", "punkt_tab")


def fmt_compact(x: float, sigfigs: int = 4) -> str:
    if not math.isfinite(x):
        return str(x)
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.{sigfigs}e}"
    return f"{x:.{sigfigs}f}"


def free_torch_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".jsonl", ".jsonl.gz"}:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        logger.info("Loaded %d records from JSONL %s", len(records), path)
        return records

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        logger.info("Loaded %d records from JSON %s", len(data), path)
        return data
    if isinstance(data, dict) and "predictions" in data:
        preds = data["predictions"]
        logger.info("Loaded %d records from JSON %s (key='predictions')", len(preds), path)
        return preds

    raise ValueError(f"Unrecognized JSON structure in {path}")


def format_triples_as_text(triples: List[Dict[str, str]]) -> str:
    bits: List[str] = []
    for t in triples:
        s = (t.get("subject") or "").strip()
        p = (t.get("predicate") or "").strip()
        o = (t.get("object") or "").strip()
        bits.append(f"({s}) --[{p}]--> ({o})")
    return " ; ".join(bits)


@dataclass
class EvalConfig:
    input_path: Path
    output_path: Optional[Path] = None
    save_per_example: Optional[Path] = None

    compute_bleu: bool = True
    compute_bertscore: bool = True
    compute_alignscore: bool = True
    compute_factspotter: bool = True
    compute_bleurt: bool = True
    compute_infolm: bool = True

    bertscore_lang: str = "en"
    bleurt_checkpoint: Optional[str] = None

    infolm_model_name: str = "bert-base-uncased"
    infolm_information_measure: str = "kl_divergence"
    infolm_idf: bool = True
    infolm_temperature: float = 0.25
    infolm_alpha: Optional[float] = None
    infolm_beta: Optional[float] = None
    infolm_batch_size: int = 64
    infolm_max_length: Optional[int] = None
    infolm_device: Optional[str] = None

    align_model: str = "roberta-base"
    align_ckpt_path: Optional[Path] = None
    align_batch_size: int = 8
    align_device: str = "cuda:0"
    align_eval_mode: str = "nli_sp"

    factspotter_model_name: str = "Inria-CEDAR/FactSpotter-DeBERTaV3-Base"
    factspotter_device: str = "cuda"
    factspotter_batch_size: int = 32
    factspotter_entailment_threshold: float = 0.499


def compute_bleu(references: List[str], predictions: List[str]) -> Dict[str, Any]:
    try:
        import sacrebleu
    except ImportError:
        logger.warning("sacrebleu is not installed; skipping BLEU.")
        return {}

    corpus_bleu = sacrebleu.corpus_bleu(predictions, [references])
    sent_scores = [
        float(sacrebleu.sentence_bleu(pred, [ref]).score)
        for pred, ref in zip(predictions, references)
    ]

    return {
        "BLEU": {
            "micro-avg": float(corpus_bleu.score),
            "sentence": {
                "mean": float(statistics.mean(sent_scores)) if sent_scores else 0.0,
                "std": float(statistics.stdev(sent_scores)) if len(sent_scores) > 1 else 0.0,
                "min": float(min(sent_scores)) if sent_scores else 0.0,
                "max": float(max(sent_scores)) if sent_scores else 0.0,
                "per_sample": sent_scores,
            },
        }
    }

def estimate_bertscore_batch_size(
        references: List[str],
        predictions: List[str],
        max_batch_size: int = 64,
        min_batch_size: int = 1,
        full_batch_max_chars: int = 256,
) -> int:
    """
    Estimate BERTScore batch size from the longest raw string length.

    Keeps batch_size=max_batch_size up to full_batch_max_chars, then scales down
    approximately inversely with length.
    """
    max_chars = max(
        (len(x or "") for x in references + predictions),
        default=0,
    )

    if max_chars <= 0:
        return max_batch_size

    scale = max(1, math.ceil(max_chars / full_batch_max_chars))
    batch_size = max_batch_size // scale

    return max(min_batch_size, min(max_batch_size, batch_size))

def compute_bertscore(references: List[str], predictions: List[str], lang: str = "en") -> Dict[str, Any]:
    try:
        from bert_score import score as bert_score  # type: ignore
    except ImportError:
        logger.warning("bert-score is not installed; skipping BERTScore.")
        return {}

    batch_size = estimate_bertscore_batch_size(
        references=references,
        predictions=predictions,
        max_batch_size=64,
        full_batch_max_chars=256,
    )

    with torch.no_grad():
        p_scores, r_scores, f1_scores = bert_score(
            predictions,
            references,
            lang=lang,
            batch_size=batch_size,
        )

    p_list = p_scores.tolist()
    r_list = r_scores.tolist()
    f1_list = f1_scores.tolist()

    def stats(xs: List[float]) -> Dict[str, Any]:
        if not xs:
            return {"mean": 0.0, "std": 0.0, "per_sample": []}
        return {
            "mean": float(statistics.mean(xs)),
            "std": float(statistics.stdev(xs)) if len(xs) > 1 else 0.0,
            "per_sample": [float(x) for x in xs],
        }

    return {
        "BERTScore": {
            "precision": stats(p_list),
            "recall": stats(r_list),
            "f1": stats(f1_list),
        }
    }


def compute_bleurt(references: List[str], predictions: List[str], cfg: EvalConfig) -> Dict[str, Any]:
    if not cfg.bleurt_checkpoint:
        logger.warning("No BLEURT checkpoint provided; skipping BLEURT.")
        return {}

    try:
        from bleurt import score as bleurt_score  # type: ignore
    except ImportError:
        logger.warning("bleurt is not installed; skipping BLEURT.")
        return {}

    logger.info("Loading BLEURT checkpoint from %s", cfg.bleurt_checkpoint)
    scorer = bleurt_score.BleurtScorer(cfg.bleurt_checkpoint)

    logger.info("Computing BLEURT on %d examples...", len(predictions))
    scores = scorer.score(references=references, candidates=predictions)
    if not scores:
        return {}

    scores = [float(s) for s in scores]
    return {
        "BLEURT": {
            "mean": float(statistics.mean(scores)),
            "std": float(statistics.stdev(scores)) if len(scores) > 1 else 0.0,
            "min": float(min(scores)),
            "max": float(max(scores)),
            "per_sample": scores,
        }
    }


def compute_infolm(references: List[str], predictions: List[str], cfg: EvalConfig) -> Dict[str, Any]:
    try:
        from torchmetrics.text.infolm import InfoLM  # type: ignore
    except ImportError:
        logger.warning("torchmetrics is not installed; skipping InfoLM.")
        return {}

    logger.info(
        "Loading InfoLM with model=%s, measure=%s",
        cfg.infolm_model_name,
        cfg.infolm_information_measure,
    )

    metric = InfoLM(
        model_name_or_path=cfg.infolm_model_name,
        temperature=cfg.infolm_temperature,
        information_measure=cfg.infolm_information_measure,
        idf=cfg.infolm_idf,
        alpha=cfg.infolm_alpha,
        beta=cfg.infolm_beta,
        device=cfg.infolm_device,
        max_length=cfg.infolm_max_length,
        batch_size=cfg.infolm_batch_size,
        return_sentence_level_score=True,
    )

    logger.info("Computing InfoLM on %d examples...", len(predictions))
    corpus_score, sentence_scores = metric(preds=predictions, target=references)

    if hasattr(corpus_score, "item"):
        corpus_score_f = float(corpus_score.item())
    else:
        corpus_score_f = float(corpus_score)

    if sentence_scores is None:
        per_sample: List[float] = []
    elif isinstance(sentence_scores, (list, tuple)):
        per_sample = [float(s.item()) if hasattr(s, "item") else float(s) for s in sentence_scores]
    elif torch.is_tensor(sentence_scores):
        per_sample = [float(x) for x in sentence_scores.view(-1).tolist()]
    else:
        per_sample = [float(sentence_scores)]

    if per_sample:
        mean = float(statistics.mean(per_sample))
        std = float(statistics.stdev(per_sample)) if len(per_sample) > 1 else 0.0
        min_v = float(min(per_sample))
        max_v = float(max(per_sample))
    else:
        mean = corpus_score_f
        std = 0.0
        min_v = corpus_score_f
        max_v = corpus_score_f

    return {
        "InfoLM": {
            "corpus": corpus_score_f,
            "mean": mean,
            "std": std,
            "min": min_v,
            "max": max_v,
            "per_sample": per_sample,
        }
    }


def compute_alignscore(contexts: List[str], claims: List[str], cfg: EvalConfig) -> Dict[str, Any]:
    if not cfg.align_ckpt_path:
        logger.warning("No AlignScore ckpt path provided; skipping AlignScore.")
        return {}

    try:
        from alignscore import AlignScore  # type: ignore
    except ImportError:
        logger.warning("alignscore is not installed; skipping AlignScore.")
        return {}

    logger.info("Loading AlignScore (model=%s, ckpt=%s)", cfg.align_model, cfg.align_ckpt_path)
    scorer = AlignScore(
        model=cfg.align_model,
        batch_size=cfg.align_batch_size,
        device=cfg.align_device,
        ckpt_path=str(cfg.align_ckpt_path),
        evaluation_mode=cfg.align_eval_mode,
    )

    logger.info("Computing AlignScore on %d examples...", len(contexts))
    scores = scorer.score(contexts=contexts, claims=claims)  # type: ignore
    scores = [float(s) for s in scores]
    if not scores:
        return {}

    return {
        "AlignScore": {
            "mean": float(statistics.mean(scores)),
            "std": float(statistics.stdev(scores)) if len(scores) > 1 else 0.0,
            "min": float(min(scores)),
            "max": float(max(scores)),
            "per_sample": scores,
        }
    }


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def compute_factspotter(
    triples_list: List[List[Dict[str, str]]],
    predictions: List[str],
    cfg: EvalConfig,
) -> Dict[str, Any]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        logger.warning("transformers/torch not installed; skipping FactSpotter.")
        return {}

    def is_cuda_oom(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return (
            isinstance(exc, RuntimeError)
            and (
                "cuda out of memory" in msg
                or "out of memory" in msg
                or "cublas_status_alloc_failed" in msg
            )
        )

    requested_device = cfg.factspotter_device
    if requested_device != "cpu" and not torch.cuda.is_available():
        requested_device = "cpu"

    device = torch.device(requested_device)

    free_torch_gpu()

    logger.info(
        "Loading FactSpotter model %s on device %s",
        cfg.factspotter_model_name,
        device,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.factspotter_model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.factspotter_model_name,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()

    try:
        model.to(device)
    except RuntimeError as exc:
        if device.type == "cuda" and is_cuda_oom(exc):
            logger.warning(
                "CUDA OOM while loading FactSpotter onto GPU. "
                "Falling back to CPU without changing tokenization, precision, "
                "threshold, or label index."
            )
            device = torch.device("cpu")
            free_torch_gpu()
            model.to(device)
        else:
            raise

    def score_batch(batch_pairs: List[Tuple[str, str]]) -> List[float]:
        nonlocal device, model

        def _score_on_current_device() -> List[float]:
            enc = tokenizer(
                batch_pairs,
                truncation=True,
                padding=True,
                return_token_type_ids=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.inference_mode():
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1)

            # Keep original FactSpotter behavior: entailment is column 0.
            out = probs[:, 0].detach().cpu().tolist()

            del enc, logits, probs
            return [float(x) for x in out]

        try:
            return _score_on_current_device()

        except RuntimeError as exc:
            if device.type == "cuda" and is_cuda_oom(exc):
                logger.warning(
                    "CUDA OOM during FactSpotter forward pass. "
                    "Moving FactSpotter to CPU and continuing with the same "
                    "fp32 model, tokenizer settings, threshold, and label index."
                )
                free_torch_gpu()
                device = torch.device("cpu")
                model.to(device)
                free_torch_gpu()
                return _score_on_current_device()

            raise

    total_triples = sum(len(t or []) for t in triples_list)
    if total_triples == 0:
        logger.warning("No triples found; skipping FactSpotter.")
        return {}

    per_example_recalls: List[float] = []
    near_threshold_count = 0
    closest_to_threshold = float("inf")

    logger.info("Computing FactSpotter scores over %d triples...", total_triples)

    pbar = tqdm.tqdm(total=total_triples, desc="FactSpotter")

    try:
        for triples, text in zip(triples_list, predictions):
            if not triples:
                continue

            n_entailed = 0
            n_total = 0

            def pairs_for_example() -> Iterable[Tuple[str, str]]:
                for t in triples:
                    triple_str = (
                        f"{t.get('subject', '')} | "
                        f"{t.get('predicate', '')} | "
                        f"{t.get('object', '')}"
                    )
                    yield text, triple_str

            for batch_pairs in batched(
                pairs_for_example(),
                max(1, cfg.factspotter_batch_size),
            ):
                batch_probs = score_batch(batch_pairs)

                for p in batch_probs:
                    margin = abs(p - cfg.factspotter_entailment_threshold)
                    closest_to_threshold = min(closest_to_threshold, margin)
                    if margin < 1e-6:
                        near_threshold_count += 1

                    if p >= cfg.factspotter_entailment_threshold:
                        n_entailed += 1

                n_total += len(batch_probs)
                pbar.update(len(batch_probs))

                if device.type == "cuda":
                    free_torch_gpu()

            if n_total > 0:
                per_example_recalls.append(float(n_entailed / float(n_total)))

    finally:
        pbar.close()
        del model, tokenizer
        free_torch_gpu()

    if near_threshold_count:
        logger.warning(
            "FactSpotter found %d probabilities within 1e-6 of the threshold. "
            "If CPU fallback was used, these borderline cases are the only ones "
            "likely to risk a CPU/GPU decision difference.",
            near_threshold_count,
        )

    if math.isfinite(closest_to_threshold):
        logger.info(
            "Closest FactSpotter probability-threshold margin: %.10g",
            closest_to_threshold,
        )

    if not per_example_recalls:
        logger.warning("No examples with triples for FactSpotter.")
        return {}

    return {
        "FactSpotter": {
            "mean": float(statistics.mean(per_example_recalls)),
            "std": float(statistics.stdev(per_example_recalls))
            if len(per_example_recalls) > 1
            else 0.0,
            "min": float(min(per_example_recalls)),
            "max": float(max(per_example_recalls)),
            "per_sample": per_example_recalls,
        }
    }


def evaluate(cfg: EvalConfig) -> Dict[str, Any]:
    records = load_predictions(cfg.input_path)

    predictions: List[str] = []
    references: List[str] = []
    triples_list: List[List[Dict[str, str]]] = []

    for rec in records:
        ref = rec.get("reference")
        if not ref:
            continue
        pred = rec.get("prediction") or "Don't know"
        triples = rec.get("triples") or rec.get("triples_parsed") or []
        predictions.append(str(pred))
        references.append(str(ref))
        triples_list.append(list(triples))

    if not predictions:
        raise ValueError("No valid predictions found in input file.")

    logger.info("Loaded %d valid examples.", len(predictions))
    overall: Dict[str, Any] = {}

    if cfg.compute_bleu and any(r.strip() for r in references):
        overall.update(compute_bleu(references, predictions))

    if cfg.compute_bertscore and any(r.strip() for r in references):
        overall.update(compute_bertscore(references, predictions, lang=cfg.bertscore_lang))
        free_torch_gpu()

    if cfg.compute_bleurt and any(r.strip() for r in references):
        overall.update(compute_bleurt(references, predictions, cfg))
        free_torch_gpu()

    if cfg.compute_infolm and any(r.strip() for r in references):
        overall.update(compute_infolm(references, predictions, cfg))
        free_torch_gpu()

    if cfg.compute_alignscore:
        contexts = [format_triples_as_text(t) for t in triples_list]
        overall.update(compute_alignscore(contexts, predictions, cfg))
        free_torch_gpu()

    if cfg.compute_factspotter:
        overall.update(compute_factspotter(triples_list, predictions, cfg))
        free_torch_gpu()

    logger.info("Overall metrics:")
    for k, v in overall.items():
        if k == "BLEU":
            logger.info("  %s = %s", k, fmt_compact(v["micro-avg"]))
        elif k == "BERTScore":
            logger.info("  %s = %s", k, fmt_compact(v["f1"]["mean"] * 100))
        elif k in {"BLEURT", "InfoLM"}:
            logger.info("  %s = %s", k, fmt_compact(v["mean"]))
        elif k in {"AlignScore", "FactSpotter"}:
            logger.info("  %s = %s", k, fmt_compact(v["mean"] * 100))
        else:
            logger.info("  %s = %s", k, fmt_compact(float(np.round(v, 2))))

    if cfg.output_path is not None:
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg.output_path.open("w", encoding="utf-8") as f:
            json.dump(overall, f, indent=2, ensure_ascii=False)
        logger.info("Saved overall metrics to %s", cfg.output_path)

    return overall


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate KG→text predictions JSON.")
    p.add_argument("input_path", type=Path, help="Path to JSON or JSONL predictions file.")
    p.add_argument("--output-path", type=Path, default=None, help="Where to write overall metrics JSON.")
    p.add_argument("--save-per-example", type=Path, default=None)

    p.add_argument("--no-bleu", action="store_true")
    p.add_argument("--no-bertscore", action="store_true")
    p.add_argument("--no-alignscore", action="store_true")
    p.add_argument("--no-factspotter", action="store_true")
    p.add_argument("--no-bleurt", action="store_true")
    p.add_argument("--no-infolm", action="store_true")

    p.add_argument("--bertscore-lang", type=str, default="en")
    p.add_argument("--bleurt-checkpoint", type=str, default="/models/BLEURT-20")

    p.add_argument("--infolm-model-name", type=str, default="bert-base-uncased")
    p.add_argument(
        "--infolm-information-measure",
        type=str,
        default="kl_divergence",
        choices=[
            "kl_divergence",
            "alpha_divergence",
            "beta_divergence",
            "ab_divergence",
            "renyi_divergence",
            "l1_distance",
            "l2_distance",
            "l_infinity_distance",
            "fisher_rao_distance",
        ],
    )
    p.add_argument("--infolm-no-idf", action="store_false", dest="infolm_idf")
    p.add_argument("--infolm-temperature", type=float, default=0.25)
    p.add_argument("--infolm-alpha", type=float, default=None)
    p.add_argument("--infolm-beta", type=float, default=None)
    p.add_argument("--infolm-batch-size", type=int, default=64)
    p.add_argument("--infolm-max-length", type=int, default=None)
    p.add_argument("--infolm-device", type=str, default=None)

    p.add_argument("--align-ckpt-path", type=Path, default=Path("/models/AlignScore-large.ckpt"))
    p.add_argument("--align-model", type=str, default="roberta-large")
    p.add_argument("--align-batch-size", type=int, default=8)
    p.add_argument("--align-device", type=str, default="cuda:0")
    p.add_argument(
        "--align-eval-mode",
        type=str,
        default="nli_sp",
        choices=["nli_sp", "nli", "bin_sp", "bin"],
    )

    p.add_argument("--factspotter-model-name", type=str, default="Inria-CEDAR/FactSpotter-DeBERTaV3-Base")
    p.add_argument("--factspotter-device", type=str, default="cuda")
    p.add_argument("--factspotter-batch-size", type=int, default=8)
    p.add_argument("--factspotter-entailment-threshold", type=float, default=0.5)

    args = p.parse_args()

    return EvalConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        save_per_example=args.save_per_example,
        compute_bleu=not args.no_bleu,
        compute_bertscore=not args.no_bertscore,
        compute_alignscore=not args.no_alignscore,
        compute_factspotter=not args.no_factspotter,
        compute_bleurt=not args.no_bleurt,
        compute_infolm=not args.no_infolm,
        bertscore_lang=args.bertscore_lang,
        bleurt_checkpoint=args.bleurt_checkpoint,
        infolm_model_name=args.infolm_model_name,
        infolm_information_measure=args.infolm_information_measure,
        infolm_idf=getattr(args, "infolm_idf", True),
        infolm_temperature=args.infolm_temperature,
        infolm_alpha=args.infolm_alpha,
        infolm_beta=args.infolm_beta,
        infolm_batch_size=args.infolm_batch_size,
        infolm_max_length=args.infolm_max_length,
        infolm_device=args.infolm_device,
        align_model=args.align_model,
        align_ckpt_path=args.align_ckpt_path,
        align_batch_size=args.align_batch_size,
        align_device=args.align_device,
        align_eval_mode=args.align_eval_mode,
        factspotter_model_name=args.factspotter_model_name,
        factspotter_device=args.factspotter_device,
        factspotter_batch_size=args.factspotter_batch_size,
        factspotter_entailment_threshold=args.factspotter_entailment_threshold,
    )


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
