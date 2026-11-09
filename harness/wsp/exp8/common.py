"""Names, volumes and images of the Experiment 8 rebuild (Qwen3.5-0.8B math editing) on Modal.

Volumes (all created on first use, never shared with other experiments):
  wsp-exp8            data: the downloaded release (hf/), the text-only export (models/Qwen3.5-0.8B-text/), the
                      benchmarks (bench/), built checkpoints (ckpt/<name>/), evaluation results (eval/<name>/),
                      uploaded edits (edits/<name>/), the vLLM compile cache. Never mounted in the editor's kernel.
  wsp-exp8-model      the editor's /model (read-only): config.json, generation_config.json, model.safetensors,
                      tokenizer.json, tokenizer_config.json, chat_template.jinja (the original's model_view35)
  wsp-exp8-practice   the editor's /practice (read-only): the released practice files
  wsp-exp8-work-<run> one per run: the editor's /work (read-write), also mounted by the forward-pass server
"""
import os

try:                       # only the Modal backend needs it (the VM backend imports this module without it)
    import modal
except ImportError:  # noqa: BLE001
    modal = None

APP_NAME = "wsp-exp8"
DATA_VOL = "wsp-exp8"
MODEL_VOL = "wsp-exp8-model"
PRACTICE_VOL = "wsp-exp8-practice"
WORK_VOL_PREFIX = "wsp-exp8-work-"
WORK_SYNC = "/mnt/work_sync"   # where the kernel sandbox mounts the run's work volume (/work itself is local disk)

HF_REPO = "Qwen/Qwen3.5-0.8B"
HF_REV = "2fc06364715b967f1860aea9cf38778875588b17"
TEXT_DIR = "/data/models/Qwen3.5-0.8B-text"      # inside containers that mount wsp-exp8 at /data
VIEW_FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
              "chat_template.jinja")
N_TENSORS = 320                                   # NAMES in the original kernel ("Tensor count: 320")

VLLM_VERSION = "0.30.0"
TRANSFORMERS_VERSION = "5.17.0"                   # the text export's config.json says transformers_version 5.17.0
GPU = "H100"

# ---- the VM backend (Modal is retired): paths on the VM, set by vm_gpu.py / vm.py through the environment
BACKEND = os.environ.get("EXP8_BACKEND", "modal")
MODEL_DIR = os.environ.get("EXP8_MODEL", "/model")              # the text export (the editor's /model)
PRACTICE_DIR = os.environ.get("EXP8_PRACTICE", "/practice")
BENCH_DIR = os.environ.get("EXP8_BENCH", "/data/bench")
WORK_REAL = os.environ.get("EXP8_WORK", "/work")                # where the run's /work really is
TMP_DIR = os.environ.get("EXP8_TMP", "/tmp")
GPU_MEM = float(os.environ.get("EXP8_GPU_MEM", "0.85"))         # VM: 0.80 (the GPU is shared)


def wpath(p):
    """the real location of a /work/... path"""
    return WORK_REAL + p[len("/work"):] if p.startswith("/work") else p


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOGS = os.path.join(ROOT, "lm_understand-logs", "exp8_math")
REMOTE_PKG = "/opt/exp8"                          # this folder inside the images

# The evaluation protocol "think35" (code/evaluate.py SAMPLING_THINK35) and the per-set samples (SAMPLES)
SAMPLING_THINK35 = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768,
                    "seed": 0}
SAMPLES = {"aime24": 16, "aime25": 16, "amc23": 16, "math500": 4, "minerva": 4, "olympiad": 4, "gsm8k": 2,
           "mmlupro1k": 1, "gpqa": 4, "ifeval": 2}
# shards per set as in the released eval/*/parts (problems i::k per job)
SHARDS = {"aime24": 3, "aime25": 3, "amc23": 4, "math500": 10, "minerva": 6, "olympiad": 12, "gsm8k": 10,
          "mmlupro1k": 5, "gpqa": 4, "ifeval": 5}
EIGHT_SETS = ("math500", "gsm8k", "amc23", "minerva", "olympiad", "mmlupro1k", "gpqa", "ifeval")

# forward-pass limits: profiles.V3_LIMITS["qwen35think"] and the first-turn prompt
LIMITS = {"gen_max": 512, "gen_max_both": 256, "tf_max_texts": 1024, "tf_max_tokens": 32768, "tf_total_tokens": 524288,
          "act_max_tokens": 8192, "max_tokens": 32768, "topk_max": 50, "attn_max_texts": 2, "attn_max_tokens": 2048,
          "tf_max_bytes": 6 * 1024 ** 3}
FULL_ATTN_LAYERS = (3, 7, 11, 15, 19, 23)
N_LAYERS = 24
FROZEN = ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight")


def data_vol():
    return modal.Volume.from_name(DATA_VOL, create_if_missing=True)


def model_vol():
    return modal.Volume.from_name(MODEL_VOL, create_if_missing=True)


def practice_vol():
    return modal.Volume.from_name(PRACTICE_VOL, create_if_missing=True)


def work_vol(run_id):
    assert run_id and all(c.isalnum() or c in "-_" for c in run_id), run_id
    return modal.Volume.from_name(WORK_VOL_PREFIX + run_id, create_if_missing=True, version=2)


def gpu_image():
    """vLLM (Qwen3.5's hybrid Gated DeltaNet support), transformers for the teacher-forced reads and activations,
    math-verify at the original's versions, the IFEval checkers' dependencies."""
    return (modal.Image.debian_slim(python_version="3.12")
            .apt_install("git")
            .uv_pip_install(f"vllm=={VLLM_VERSION}", f"transformers=={TRANSFORMERS_VERSION}",
                         "math-verify==0.9.0", "latex2sympy2_extended==1.11.0", "antlr4-python3-runtime==4.13.2",
                         "langdetect==1.0.9", "immutabledict==4.3.1", "nltk==3.10.4",
                         "flash-linear-attention==0.6.0")
            .run_commands("python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab')\"")
            .env({"TOKENIZERS_PARALLELISM": "false",
                  "HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
                  "VLLM_USE_FLASHINFER_SAMPLER": "0"})
            .add_local_dir(HERE, REMOTE_PKG, ignore=["__pycache__", "*.pyc", "eval_results"]))


def cpu_image():
    return (modal.Image.debian_slim(python_version="3.12")
            .pip_install("numpy==2.2.6", "huggingface_hub==2.4.0", "datasets==5.1.0", "safetensors==0.5.3",
                         "torch==2.7.1", index_url=None, extra_index_url="https://download.pytorch.org/whl/cpu")
            .add_local_dir(HERE, REMOTE_PKG, ignore=["__pycache__", "*.pyc", "eval_results"])
            .add_local_dir(os.path.join(LOGS, "base_model", "Qwen3.5-0.8B-text"), "/opt/base_model_text")
            .add_local_dir(os.path.join(LOGS, "practice"), "/opt/practice"))


def kernel_image():
    """The editor's kernel: CPU only, the packages the first-turn prompt lists."""
    return (modal.Image.debian_slim(python_version="3.10")
            .apt_install("rsync")
            .pip_install("numpy==2.2.6", "scipy==1.15.3", "scikit-learn==1.6.1", "pandas==2.2.3", "sympy==1.13.3",
                         "networkx==3.4.2", "safetensors==0.5.3", "tokenizers==0.22.1")
            .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cpu")
            .env({"OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16", "CUDA_VISIBLE_DEVICES": "",
                  "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
            .add_local_file(os.path.join(HERE, "kernel_server.py"), "/opt/kernel/kernel_server.py")
            .add_local_file(os.path.join(HERE, "kernel_prelude.py"), "/opt/kernel/kernel_prelude.py"))
