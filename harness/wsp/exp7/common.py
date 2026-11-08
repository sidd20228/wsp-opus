"""Names, paths and images of the Experiment 7 rebuild (Gemma 3 270M IT honest / sycophantic pairs) on Modal.

Volume `wsp-exp7` (mounted at /data), laid out like the original's SYCMID_ROOT so that the released scripts run
unchanged with SYCMID_ROOT=/data/sycmid:
  /data/sycmid/models/gemma-3-270m-it/          the starting model (snapshot of google/gemma-3-270m-it, BASE_REV)
  /data/sycmid/private/splits/pair<P>.json      the released splits (copied verbatim from dataset/splits/)
  /data/sycmid/private/runs/pair<P>/<role>/     train.py output: log.jsonl, split.json, final/ (the checkpoint)
  /data/sycmid/private/truth/pair<P>_<m>.json   eval.py output (m = honest, sycophant, base)
  /data/sycmid/release_dev/pair_<hex>/          make_release.py dev    (layers only, no tokenizer; rounds 1-3)
  /data/sycmid/release_devemb/pair_<hex>/       make_release.py devemb (all tensors + tokenizer; round 4, round 5)
  /data/sycmid/private/release_<set>_map.json   make_release.py's A/B map (checked against eval/answer_key/)
  /data/sycmid/meta/                            base revision, timings, sha256 of every file
"""
import os

import modal

APP_NAME = "wsp-exp7"
VOL_NAME = "wsp-exp7"
SECRET_NAME = "wsp-hf"                       # env HF_TOKEN; never printed or written
GPU = ["L4", "A10G"]           # either; the recipe needs <10 GB

BASE_REPO = "google/gemma-3-270m-it"
ROOT_REMOTE = "/data/sycmid"
BASE_REMOTE = f"{ROOT_REMOTE}/models/gemma-3-270m-it"
PAIRS = (100, 101, 102)
ROLES = ("honest", "sycophant")
SETS = {"dev": "layers only, no tokenizer (rounds 1-3)", "devemb": "all tensors + tokenizer (round 4, round 5)"}

TORCH_VERSION = "2.9.1"
TRANSFORMERS_VERSION = "5.17.0"              # the released checkpoints' config.json: "transformers_version": "5.17.0"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOGS = os.path.join(ROOT, "lm_understand-logs", "exp7_sycophancy")
LOCAL_MODELS = os.path.join(ROOT, "models", "exp7")
LOCAL_RELEASES = os.path.join(ROOT, "runs", "opus55", "exp7_sycophancy", "releases")
REMOTE_PKG = "/opt/exp7"                     # the released code/ and dataset/ (verbatim) and this folder


def volume():
    return modal.Volume.from_name(VOL_NAME, create_if_missing=True)


def image():
    """torch + transformers 5.17.0 (the version the original checkpoints were saved with); the released code/ and
    dataset/ folders are copied in byte for byte."""
    return (modal.Image.debian_slim(python_version="3.12")
            .uv_pip_install(f"torch=={TORCH_VERSION}", f"transformers=={TRANSFORMERS_VERSION}", "numpy==2.2.6",
                            "safetensors", "huggingface_hub", "accelerate")
            .env({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
                  "PYTHONUNBUFFERED": "1"})
            .add_local_dir(os.path.join(LOGS, "code"), f"{REMOTE_PKG}/code")
            .add_local_dir(os.path.join(LOGS, "dataset"), f"{REMOTE_PKG}/dataset")
            .add_local_dir(HERE, f"{REMOTE_PKG}/wsp_exp7", ignore=["__pycache__", "*.pyc"]))
