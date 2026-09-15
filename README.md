# Metrics API for KG-to-Text Evaluation

Dockerized HTTP API for the evaluation metrics used in our long-form KG-to-text experiments. The service exposes a single interface for **BLEU**, **BERTScore**, **BLEURT**, **AlignScore**, and **FactSpotter**.


```bash
docker pull tchewik/metrics-api:1.0.0

docker run --rm --gpus all \
  -p 8125:8000 \
  tchewik/metrics-api:1.0.0
```

Or use Docker Compose:

```bash
docker compose up -d
```

The API is then available at `http://localhost:8125`.

```bash
curl http://localhost:8125/healthz
curl http://localhost:8125/v1/metrics
```

## Models included in the image

The build copies local artifacts into fixed runtime locations:

- `models/AlignScore-large.ckpt` → `/models/AlignScore-large.ckpt`
- `models/BLEURT-20/` → `/models/BLEURT-20/`
- `cache/huggingface/` → `/cache/huggingface/`

The default metric set needs these Hugging Face models already present in `cache/huggingface/`:

- `roberta-large` for English BERTScore and AlignScore
- `Inria-CEDAR/FactSpotter-DeBERTaV3-Base` for FactSpotter


## Remote evaluation client

A predictions file can be evaluated with the included client:

```bash
python client/remote_eval_client.py predictions.jsonl \
  --api-url https://<HOST>/v1/evaluate/file \
  --output-path eval_metrics.json \
  --metrics bleu,bertscore,bleurt,alignscore,factspotter
```

Only the lightweight `requests` dependency is needed to use the client outside the container:

```bash
python -m pip install requests
```

For multi-seed experiments:

```bash
python client/aggregate_multi_seed_metrics_remote.py \
  --seeds 42,43,44 \
  --base_output_dir runs/my_experiment \
  --pred_relpath direct/test_predictions.jsonl \
  --api-url https://<HOST>/v1/evaluate/file \
  --metrics bleu,bertscore,bleurt,alignscore,factspotter
```

If you need the normalized aggregate score as well (Avg), pass a compatible normalization-range JSON file with `--norm_ranges /path/to/metric_ranges.json`.

## Input format

The file endpoint accepts JSON or JSONL records. A typical JSONL row is:

```json
{
  "idx": 1,
  "triples": [
    {
      "subject": "Aarhus Airport",
      "predicate": "city served",
      "object": "Aarhus, Denmark"
    }
  ],
  "reference": "Aarhus Airport serves Aarhus in Denmark.",
  "prediction": "Aarhus Airport is the airport of Aarhus, Denmark."
}
```

Reference-based metrics use `reference` and `prediction`; graph-grounded evaluation uses `triples` together with `prediction`.

## Evaluate a file with curl

```bash
curl -X POST http://localhost:8125/v1/evaluate/file \
  -F 'file=@predictions.jsonl' \
  -F 'metrics=bleu,bertscore,bleurt,alignscore,factspotter'
```

## Evaluate raw records

```bash
curl -X POST http://localhost:8125/v1/evaluate/records \
  -H 'Content-Type: application/json' \
  -d '{
    "records": [
      {
        "idx": 0,
        "triples": [],
        "reference": "hello",
        "prediction": "hello"
      }
    ],
    "metrics": ["bertscore", "bleurt", "alignscore"]
  }'
```

## API endpoints

| Endpoint | Method | Purpose |
|---|---:|---|
| `/healthz` | GET | Service and GPU visibility status |
| `/v1/metrics` | GET | Supported metrics and defaults |
| `/v1/evaluate/file` | POST | Evaluate an uploaded JSON/JSONL file |
| `/v1/evaluate/records` | POST | Evaluate records supplied in JSON |
| `/v1/jobs` | GET | Recent job information |
| `/v1/jobs/{request_id}` | GET | Information for one evaluation job |

FastAPI's interactive OpenAPI documentation is available at `http://localhost:8125/docs`.

## Supported metrics

The service currently supports:

- `bleu`
- `bertscore`
- `bleurt`
- `alignscore`
- `factspotter`

The default paper-oriented set is:

```text
bleu,bertscore,bleurt,alignscore,factspotter
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ALIGN_CKPT_PATH` | `/models/AlignScore-large.ckpt` | Bundled AlignScore checkpoint path |
| `BLEURT_CHECKPOINT` | `/models/BLEURT-20` | Bundled BLEURT checkpoint directory |
| `HF_HOME` | `/cache/huggingface` | Bundled Hugging Face cache root |
| `TRANSFORMERS_CACHE` | `/cache/huggingface` | Bundled Transformers cache root |
| `BUNDLED_HF_MODELS` | `roberta-large;Inria-CEDAR/FactSpotter-DeBERTaV3-Base` | Semicolon-separated HF repos verified locally |
| `NLTK_DATA` | `/cache/nltk_data` | NLTK data baked during the image build |
| `SERVICE_GPU_DEVICE` | `cuda:0` | Device used by AlignScore |
| `SERVICE_FACTSPOTTER_DEVICE` | `cuda` | Device used by FactSpotter |
| `METRICS_API_JOB_TIMEOUT_SEC` | `300` in the app; `7200` in Compose | Evaluation timeout |
| `MAX_UPLOAD_MB` | `256` | Maximum upload size |
