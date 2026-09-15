FROM python:3.11-slim AS build

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG ALIGN_REPO_URL=https://github.com/tchewik/AlignScore.git
ARG ALIGN_GIT_REF=main
ARG BLEURT_REPO_URL=https://github.com/google-research/bleurt.git
ARG BLEURT_GIT_REF=master

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/metrics-api
COPY requirements-core.txt requirements-bleurt.txt ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio \
    && python -m pip install -r requirements-core.txt \
    && python -m pip install -r requirements-bleurt.txt \
    && python -m pip install nltk pytorch_lightning

RUN git clone "${ALIGN_REPO_URL}" /opt/AlignScore \
    && git -C /opt/AlignScore checkout "${ALIGN_GIT_REF}" \
    && python -m pip install /opt/AlignScore \
    && python -m spacy download en_core_web_sm \
    && git clone "${BLEURT_REPO_URL}" /opt/bleurt \
    && git -C /opt/bleurt checkout "${BLEURT_GIT_REF}" \
    && python -m pip install /opt/bleurt \
    && mkdir -p /opt/nltk_data \
    && python - <<'PY'
import nltk
for package in ("punkt", "punkt_tab"):
    nltk.download(package, download_dir="/opt/nltk_data", quiet=True, raise_on_error=True)
PY

FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    NLTK_DATA=/cache/nltk_data \
    TOKENIZERS_PARALLELISM=false \
    SERVICE_GPU_DEVICE=cuda:0 \
    SERVICE_FACTSPOTTER_DEVICE=cuda \
    MAX_UPLOAD_MB=256 \
    ALIGN_CKPT_PATH=/models/AlignScore-large.ckpt \
    BLEURT_CHECKPOINT=/models/BLEURT-20 \
    BUNDLED_HF_MODELS="roberta-large;Inria-CEDAR/FactSpotter-DeBERTaV3-Base" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/metrics-api

COPY --from=build /usr/local /usr/local
COPY --from=build /opt/nltk_data/ /cache/nltk_data/

COPY models/ /models/
COPY cache/huggingface/ /cache/huggingface/

COPY scripts/verify_bundled_models.py /usr/local/bin/verify-bundled-models
RUN chmod +x /usr/local/bin/verify-bundled-models \
    && /usr/local/bin/verify-bundled-models

COPY app ./app
COPY vendor ./vendor
COPY client ./client
COPY docker-entrypoint.sh /usr/local/bin/metrics-api-entrypoint
RUN chmod +x /usr/local/bin/metrics-api-entrypoint

ENV PYTHONPATH=/opt/metrics-api/vendor:/opt/metrics-api

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/metrics-api-entrypoint"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
