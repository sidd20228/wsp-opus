"""The Modal app `wsp-exp8`: model download and export, benchmark preparation, checkpoint building, the forward-pass
server and the benchmark evaluation. Deploy once (every function scales to zero):

    .venv/bin/modal deploy wsp/exp8/app.py

Local callers (wsp/exp8/fwd.py, wsp/exp8/evaluate.py, wsp/exp8/kernel.py) look the functions up by name.
"""
import json
import os
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/opt/exp8")
import common as C  # noqa: E402

app = modal.App(C.APP_NAME)
DATA = C.data_vol()
MODEL = C.model_vol()
PRACTICE = C.practice_vol()
GPU_IMAGE = C.gpu_image()
CPU_IMAGE = C.cpu_image()


def _pkg():
    if C.REMOTE_PKG not in sys.path:
        sys.path.insert(0, C.REMOTE_PKG)


# ------------------------------------------------------------------------------------------------------- setup
@app.function(image=CPU_IMAGE, volumes={"/data": DATA, "/view_model": MODEL, "/view_practice": PRACTICE},
              timeout=3600, cpu=4, memory=16384)
def setup_model(force: bool = False):
    """Download Qwen/Qwen3.5-0.8B at the pinned revision, build the text-only export (the original's
    models/Qwen3.5-0.8B-text: Qwen3_5ForCausalLM, one model.safetensors with the model.language_model.* tensors, the
    released text config / generation config / tokenizer config / chat template, the release's tokenizer.json) and
    fill the editor's read-only /model and /practice volumes."""
    import hashlib
    import shutil
    from huggingface_hub import snapshot_download
    _pkg()
    from vendor import st_numpy as S
    hf = "/data/hf/Qwen3.5-0.8B"
    if force or not os.path.exists(os.path.join(hf, "config.json")):
        snapshot_download(C.HF_REPO, revision=C.HF_REV, local_dir=hf)
    src = [f for f in os.listdir(hf) if f.endswith(".safetensors")]
    assert len(src) == 1, src
    out = C.TEXT_DIR
    os.makedirs(out, exist_ok=True)
    tensors = S.load(os.path.join(hf, src[0]))
    text = {k: v for k, v in tensors.items() if k.startswith("model.language_model.")}
    others = sorted({k.split(".")[0] + "." + k.split(".")[1] for k in tensors if k not in text})
    assert len(text) == C.N_TENSORS, (len(text), others)
    S.save(os.path.join(out, "model.safetensors"), text, {"format": "pt"})
    for f in ("config.json", "generation_config.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copyfile(os.path.join("/opt/base_model_text", f), os.path.join(out, f))
    shutil.copyfile(os.path.join(hf, "tokenizer.json"), os.path.join(out, "tokenizer.json"))
    DATA.commit()
    for f in C.VIEW_FILES:
        shutil.copyfile(os.path.join(out, f), os.path.join("/view_model", f))
    MODEL.commit()
    for f in os.listdir("/opt/practice"):
        shutil.copyfile(os.path.join("/opt/practice", f), os.path.join("/view_practice", f))
    PRACTICE.commit()
    sha = hashlib.sha256(open(os.path.join(out, "model.safetensors"), "rb").read()).hexdigest()
    return {"text_tensors": len(text), "other_prefixes": others, "model_safetensors_sha256": sha,
            "files": sorted(os.listdir(out)), "practice": sorted(os.listdir("/view_practice"))}


@app.function(image=CPU_IMAGE, volumes={"/data": DATA}, timeout=3600, cpu=4, memory=16384)
def prepare_bench(order: dict, sets: list):
    """bench/<set>.jsonl like the original `evaluate.py prepare` (and the beyond-math sets), rows in the original's
    order (reconstructed from the released per-problem files of every shard and checked against them)."""
    from datasets import load_dataset
    _pkg()
    import mcprompt
    os.makedirs("/data/bench", exist_ok=True)
    rows = {}
    if "aime24" in sets:
        rows["aime24"] = [{"id": f"aime24_{r['id']}", "problem": r["problem"], "answer": str(r["answer"])}
                          for r in load_dataset("HuggingFaceH4/aime_2024", split="train")]
    if "aime25" in sets:
        rows["aime25"] = [{"id": f"aime25_{r['problem_idx']}", "problem": r["problem"], "answer": str(r["answer"])}
                          for r in load_dataset("MathArena/aime_2025", split="train")]
    if "amc23" in sets:
        rows["amc23"] = [{"id": f"amc23_{r['id']}", "problem": r["question"], "answer": str(r["answer"])}
                         for r in load_dataset("math-ai/amc23", split="test")]
    if "math500" in sets:
        rows["math500"] = [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]}
                           for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]
    if "minerva" in sets:
        rows["minerva"] = [{"id": f"minerva_{i}", "problem": r["question"], "answer": str(r["answer"])}
                           for i, r in enumerate(load_dataset("math-ai/minervamath", split="test"))]
    if "olympiad" in sets:
        rs = []
        for r in load_dataset("math-ai/olympiadbench", split="test"):
            fa = r["final_answer"]
            fa = fa if isinstance(fa, list) else json.loads(fa.replace("'", '"')) if fa.startswith("[") else [fa]
            rs.append({"id": f"olympiad_{r['id']}", "problem": r["question"], "answer": ", ".join(str(x) for x in fa)})
        rows["olympiad"] = rs
    if "gsm8k" in sets:
        rows["gsm8k"] = [{"id": f"gsm8k_test_{i}", "problem": r["question"],
                          "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                         for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test"))]
    if "mmlupro1k" in sets:
        rows["mmlupro1k"] = [{"id": f"mmlupro_{r['question_id']}", "kind": "mc", "answer": r["answer"],
                              "category": r["category"], "prompt": mcprompt.mmlu_pro_prompt(r)}
                             for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test")]
    if "ifeval" in sets:
        rows["ifeval"] = [{"id": f"ifeval_{r['key']}", "kind": "ifeval", "prompt": r["prompt"],
                           "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]}
                          for r in load_dataset("google/IFEval", split="train")]
    if "gpqa" in sets:      # gated on Hugging Face: the token comes from the Modal secret wsp-hf (prepare_gpqa)
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("gpqa: Idavidrein/gpqa is gated; call prepare_gpqa (uses the secret wsp-hf)")
        rows["gpqa"] = [mcprompt.gpqa_row(r) for r in load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")]
    report = {}
    for name, rs in rows.items():
        want = order[name]
        by = {r["id"]: r for r in rs}
        missing = [i for i in want if i not in by]
        assert not missing, (name, missing[:5])
        kept = [by[i] for i in want]
        with open(f"/data/bench/{name}.jsonl", "w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report[name] = {"rows": len(kept), "source_rows": len(rs)}
    DATA.commit()
    return report


@app.function(image=CPU_IMAGE, volumes={"/data": DATA}, timeout=3600, cpu=4, memory=16384,
              secrets=[modal.Secret.from_name("wsp-hf")])
def prepare_gpqa(order: dict):
    """GPQA Diamond (gated): the same as prepare_bench with HF_TOKEN from the Modal secret wsp-hf (never printed or
    stored)."""
    return prepare_bench.local(order, ["gpqa"])


# ------------------------------------------------------------------------------------------------- checkpoints
@app.function(image=CPU_IMAGE, volumes={"/data": DATA}, timeout=3600, cpu=4, memory=32768)
def build_ckpt(name: str, spec: dict):
    """/data/ckpt/<name>/: the text export with edited tensors.
    spec: {"kind": "base"} | {"kind": "edit_diff", "dir": <data path with edit.npz + manifest.json>} (rebuilt with the
    released apply_edit.py, every tensor hash-checked) | {"kind": "npy_dir", "dir": <data path with <name>.npy>} (a
    submission: cast to bfloat16 like build_ckpt.py) | {"kind": "work", "run": <run id>} (the run's /work/edited)."""
    import shutil
    import numpy as np
    import torch
    _pkg()
    from vendor import apply_edit, st_numpy as S
    DATA.reload()
    out = f"/data/ckpt/{name}"
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for f in C.VIEW_FILES:
        if f != "model.safetensors":
            shutil.copyfile(os.path.join(C.TEXT_DIR, f), os.path.join(out, f))
    base = os.path.join(C.TEXT_DIR, "model.safetensors")
    info = {"name": name, "spec": spec}
    if spec["kind"] == "base":
        shutil.copyfile(base, os.path.join(out, "model.safetensors"))
        info["edited"] = []
    elif spec["kind"] == "edit_diff":
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            apply_edit.main(base, spec["dir"], os.path.join(out, "model.safetensors"))
        info["apply_edit_log"] = buf.getvalue()
        info["edited"] = [t["name"] for t in json.load(open(os.path.join(spec["dir"], "manifest.json")))["tensors"]]
    else:
        if spec["kind"] == "work":
            wv = C.work_vol(spec["run"])
            src = f"/tmp/sub_{name}"
            os.makedirs(src, exist_ok=True)
            for e in wv.listdir("edited"):
                if e.path.endswith(".npy"):
                    with open(os.path.join(src, os.path.basename(e.path)), "wb") as f:
                        wv.read_file_into_fileobj(e.path, f)
        else:
            src = spec["dir"]
        edits = {f[:-4]: os.path.join(src, f) for f in os.listdir(src) if f.endswith(".npy")}
        tensors = S.load(base)
        changed = {}
        for n, p in edits.items():
            assert n in tensors, n
            dt, bits = tensors[n]
            a = torch.from_numpy(np.load(p)).to(torch.bfloat16)          # build_ckpt.py: cast to bfloat16
            assert list(a.shape) == list(bits.shape), (n, a.shape, bits.shape)
            nb = a.view(torch.int16).numpy().view(np.uint16)
            changed[n] = int((nb != bits).sum())
            tensors[n] = ("BF16", nb)
        S.save(os.path.join(out, "model.safetensors"), tensors, {"format": "pt"})
        info["edited"] = sorted(edits)
        info["changed_entries"] = changed
    json.dump(info, open(os.path.join(out, "edits.json"), "w"), indent=1)
    DATA.commit()
    return info


# --------------------------------------------------------------------------------------------- evaluation jobs
@app.function(image=GPU_IMAGE, gpu=C.GPU, volumes={"/data": DATA}, timeout=6 * 3600, cpu=8, memory=65536,
              max_containers=40, single_use_containers=True)     # one vLLM engine per container
def eval_shard(ckpt_name: str, out_name: str, set_name: str, shard: str, protocol: str = "think35"):
    """One evaluation job: the released evaluate.run for one set and one shard (problems i::k), results in
    /data/eval/<out_name>/parts/<set>_s<i>of<k>/."""
    t0 = time.time()
    _pkg()
    import evaljob
    DATA.reload()
    i, k = shard.split("/")
    out = f"/data/eval/{out_name}/parts/{set_name}_s{i}of{k}"
    res = evaljob.run(f"/data/ckpt/{ckpt_name}", out, [set_name], protocol=protocol, shard=shard)
    DATA.commit()
    return {"scores": res, "wall_s": time.time() - t0, "set": set_name, "shard": shard}


# ----------------------------------------------------------------------------------------- forward-pass server
@app.cls(image=GPU_IMAGE, gpu=C.GPU, volumes={"/data": DATA, "/model": MODEL.read_only(),
                                                "/practice": PRACTICE.read_only()},
         timeout=2 * 3600, cpu=8, memory=65536, max_containers=8,
         single_use_containers=True)     # a fresh container (one vLLM engine) per pass and model: no engine swaps
class FwdServer:
    """GPU side of the editor's forward-pass tool. Mount the run's work volume at /work with
    FwdServer.with_options(volumes={... the class volumes ..., "/work": work_vol(run)}) (fwd.server_for). Methods return
    plain dicts; fwd.py builds the summary."""

    @modal.enter()
    def start(self):
        _pkg()
        import fwdserver
        self.s = fwdserver.Server()

    @modal.method()
    def generate(self, run: str, req: dict, k: int, which: str, tag: str):
        """which: original | edited; tag: the file stem (gen_<k> or gen_<k>_<which>)."""
        return self.s.generate(run, req, k, which, tag)

    @modal.method()
    def teacher_forced(self, run: str, req: dict, k: int):
        return self.s.teacher_forced(run, req, k)

    @modal.method()
    def ping(self):
        return self.s.info()


# ------------------------------------------------------------------------------------------------- development
@app.function(image=GPU_IMAGE, gpu=C.GPU, volumes={"/data": DATA}, timeout=3600, cpu=8, memory=65536,
              single_use_containers=True)
def gpu_probe(code: str):
    """development helper: run a snippet on the GPU image (returns the value of `result`)."""
    _pkg()
    ns = {}
    exec(code, ns)
    return ns.get("result")
