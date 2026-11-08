"""GPU side of the editor's forward-pass tool (runs in app.FwdServer, one H100 per container).

generate(run, req, k, which, tag): vLLM, the evaluation's sampling (think35) and chat template (thinking on), the
  item's seed fixed by the problem (sha256 of its id, or of the prompt text for custom prompts), the run's current
  edits from /work/edited cast to bfloat16 (as build_ckpt.py) when which == "edited"; answers graded with the released
  mathqa (last \\boxed{}, math-verify); /work/<tag>.jsonl; optional activations with HF transformers (residual stream
  at the requested layers, 0 = embeddings, l = after layer l as HF's hidden_states[l]; MLP down_proj inputs of the
  requested layers < 24) at the prompt's last token and averaged over the answer's first 8,192 tokens, in
  /work/<tag>_acts/.
teacher_forced(run, req, k): HF transformers in bfloat16, one text at a time; token log-probs, top-k, hidden states,
  MLP activations (down_proj inputs) and attention probabilities of full-attention layers (eager attention), saved
  right-padded as .npy with meta.json in /work/fwd_<k>/.
Errors raise; fwd.py turns them into a failed (uncounted) pass.
"""
import gc
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from vendor import mathqa as Q  # noqa: E402

MODEL_DIR = "/model"
PREFIX = "model.language_model."


def item_seed(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") & 0x7FFFFFFF


class Server:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.practice = {}
        for f in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
            for line in open(os.path.join("/practice", f)):
                r = json.loads(line)
                self.practice[r["id"]] = r
        self.llm, self.llm_key = None, None
        self.hf = None
        self.shapes = None

    def info(self):
        import torch
        import vllm
        import transformers
        return {"vllm": vllm.__version__, "transformers": transformers.__version__, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0), "practice": len(self.practice)}

    # ------------------------------------------------------------------------------------------------- edits
    def _base_shapes(self):
        if self.shapes is None:
            from vendor import st_numpy as S
            head, _ = S.read_header(os.path.join(MODEL_DIR, "model.safetensors"))
            self.shapes = {k: tuple(v["shape"]) for k, v in head.items() if k != "__metadata__"}
        return self.shapes

    def read_edits(self, run):
        """{name: float32 array} of /work/edited after reloading the run's volume; checks the out-of-scope rules."""
        C.work_vol(run).reload()
        d = "/work/edited"
        edits = {}
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".npy"):
                    edits[f[:-4]] = np.load(os.path.join(d, f))
        shapes = self._base_shapes()
        for n, a in edits.items():
            if n in C.FROZEN:
                raise ValueError(f"edit of {n} refused: the final norm and the embedding / output layer are out of scope")
            if n not in shapes:
                raise ValueError(f"/work/edited/{n}.npy: unknown tensor name")
            if tuple(a.shape) != shapes[n]:
                raise ValueError(f"/work/edited/{n}.npy: shape {tuple(a.shape)} differs from {shapes[n]}")
            if not np.isfinite(a).all():
                raise ValueError(f"/work/edited/{n}.npy has non-finite values")
        if edits:                                     # save_edit's whole-tensor-rescale rule, checked again here
            from vendor import st_numpy as S
            base = S.load(os.path.join(MODEL_DIR, "model.safetensors"), set(edits))
            for n, a in edits.items():
                b = S.bf16_to_f32(base[n][1]).astype(np.float64).ravel()
                x = a.astype(np.float64).ravel()
                den = float(b @ b)
                if den > 0:
                    c = float(x @ b) / den
                    if np.linalg.norm(x - c * b) / den ** 0.5 < 1e-6 and abs(c - 1) > 1e-6:
                        raise ValueError(f"edit of {n} refused: it only multiplies the whole tensor by a constant")
        return edits

    @staticmethod
    def fingerprint(edits):
        h = hashlib.sha256()
        for n in sorted(edits):
            h.update(n.encode())
            h.update(np.ascontiguousarray(edits[n], dtype=np.float32).tobytes())
        return h.hexdigest()[:16]

    @staticmethod
    def _bf16_bits(a):
        import torch
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)

    def build_ckpt(self, edits, fp):
        """the text export with the edited tensors cast to bfloat16 (build_ckpt.py), in local /tmp."""
        from vendor import st_numpy as S
        out = f"/tmp/ckpt_{fp}"
        if os.path.exists(os.path.join(out, "model.safetensors")):
            return out
        os.makedirs(out, exist_ok=True)
        for f in C.VIEW_FILES:
            if f != "model.safetensors":
                shutil.copyfile(os.path.join(MODEL_DIR, f), os.path.join(out, f))
        tensors = S.load(os.path.join(MODEL_DIR, "model.safetensors"))
        for n, a in edits.items():
            tensors[n] = ("BF16", self._bf16_bits(a).reshape(tensors[n][1].shape))
        S.save(os.path.join(out, "model.safetensors"), tensors, {"format": "pt"})
        return out

    # -------------------------------------------------------------------------------------------------- vLLM
    def _drop_llm(self):
        if self.llm is not None:
            try:
                self.llm.llm_engine.engine_core.shutdown()
            except Exception:  # noqa: BLE001
                pass
            del self.llm
            self.llm, self.llm_key = None, None
            gc.collect()
            import torch
            torch.cuda.empty_cache()

    def get_llm(self, edits):
        from vllm import LLM
        key = self.fingerprint(edits) if edits else "original"
        if self.llm_key != key:
            self._drop_llm()
            path = MODEL_DIR if not edits else self.build_ckpt(edits, key)
            self.llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=0.78,
                           max_model_len=C.SAMPLING_THINK35["max_tokens"] + 2048, seed=0)
            self.llm_key = key
        return self.llm

    # ---------------------------------------------------------------------------------------------- HF model
    def get_hf(self, edits):
        """the HF model (bfloat16, GPU) with `edits` applied in place (originals kept on the CPU to restore)."""
        import torch
        from transformers import AutoModelForCausalLM
        if self.hf is None:
            self.hf = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).cuda().eval()
            self.hf_params = dict(self.hf.named_parameters())
            self.hf_saved = {}
        for n in list(self.hf_saved):                  # restore the original weights first
            self.hf_params[n].data.copy_(self.hf_saved.pop(n))
        for n, a in edits.items():
            hn = "model." + n[len(PREFIX):]
            p = self.hf_params[hn]
            self.hf_saved[hn] = p.data.detach().cpu().clone()
            bits = torch.from_numpy(self._bf16_bits(a).view(np.int16)).view(torch.bfloat16)
            p.data.copy_(bits.reshape(p.shape).to(p.device))
        return self.hf

    def _set_attn(self, impl):
        m = self.hf
        try:
            m.set_attn_implementation(impl)
        except Exception:  # noqa: BLE001
            m.config._attn_implementation = impl

    # -------------------------------------------------------------------------------------------- generation
    def _items(self, req):
        probs = req.get("problems") or []
        unknown = [p for p in probs if p not in self.practice]
        if unknown:
            raise ValueError(f"unknown practice ids: {unknown[:5]}")
        items = [{"id": p, "source": self.practice[p]["source"], "answer": self.practice[p]["answer"],
                  "user": Q.build_prompt(self.practice[p]["problem"]), "seed_key": p} for p in probs]
        for i, t in enumerate(req.get("prompts") or []):
            if not isinstance(t, str) or not t:
                raise ValueError("prompts must be non-empty strings")
            items.append({"id": f"prompt_{i}", "source": "prompt", "answer": None, "user": t, "seed_key": "prompt:" + t,
                          "prompt": t})
        return items

    def generate(self, run, req, k, which, tag):
        import torch
        from vllm import SamplingParams
        t0 = time.time()
        edits = self.read_edits(run) if which == "edited" else {}
        if which != "edited":
            C.work_vol(run).reload()
        items = self._items(req)
        n = int(req.get("n", 1))
        max_tokens = int(req.get("max_tokens", C.LIMITS["max_tokens"]))
        prompts = [self.tok.apply_chat_template([{"role": "user", "content": it["user"]}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=True) for it in items]
        samp = {kk: v for kk, v in C.SAMPLING_THINK35.items() if kk not in ("max_tokens", "seed")}
        params = [SamplingParams(n=n, max_tokens=max_tokens, seed=item_seed(it["seed_key"]), **samp) for it in items]
        llm = self.get_llm(edits)
        t_gen = time.time()
        outs = llm.generate(prompts, params)
        t_gen = time.time() - t_gen
        bad = sum(1 for o in outs for x in o.outputs if x.finish_reason not in ("stop", "length"))
        if bad:
            raise RuntimeError(f"{bad} generations ended without finishing or reaching max_tokens")
        recs, compact = [], []
        for it, o in zip(items, outs):
            samples, cs = [], []
            for x in o.outputs:
                pred, _ = Q.extract(x.text)
                correct = Q.is_correct(pred, it["answer"]) if it["answer"] is not None else None
                cut = x.finish_reason == "length"
                samples.append({"text": x.text, "pred": pred, "correct": correct, "tokens": len(x.token_ids), "cut": cut})
                cs.append({"correct": correct, "cut": cut, "tokens": len(x.token_ids), "has_pred": pred is not None})
            rec = {"id": it["id"], "source": it["source"], "answer": it["answer"], "samples": samples}
            if "prompt" in it:
                rec["prompt"] = it["prompt"]
            recs.append(rec)
            compact.append({"id": it["id"], "source": it["source"], "graded": it["answer"] is not None, "samples": cs})
        path = f"/work/{tag}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        acts = None
        if req.get("activations"):
            acts = self._activations(req["activations"], items, prompts, outs, compact, edits, f"/work/{tag}_acts")
        C.work_vol(run).commit()
        torch.cuda.empty_cache()
        return {"records": compact, "file": path, "acts": acts, "s": round(time.time() - t0, 1),
                "gen_s": round(t_gen, 1), "edited_tensors": sorted(edits)}

    def _activations(self, spec, items, prompts, outs, compact, edits, out_dir):
        import torch
        layers = [int(x) for x in spec.get("layers", [])]
        mlp = bool(spec.get("mlp", False))
        mlp_layers = [l for l in layers if l < C.N_LAYERS] if mlp else []
        m = self.get_hf(edits)
        self._set_attn("sdpa")
        cap = C.LIMITS["act_max_tokens"]
        rows = [(i, j) for i, o in enumerate(outs) for j in range(len(o.outputs))]
        N = len(rows)
        H = m.config.hidden_size
        I = m.config.intermediate_size
        hl = np.zeros((N, len(layers), H), np.float32)
        hm = np.zeros((N, len(layers), H), np.float32)
        ml = np.zeros((N, len(mlp_layers), I), np.float32)
        mm = np.zeros((N, len(mlp_layers), I), np.float32)
        store = {}
        hooks = [m.model.layers[l].mlp.down_proj.register_forward_pre_hook(
            (lambda l: lambda mod, args: store.__setitem__(l, args[0]))(l)) for l in mlp_layers]
        index = []
        try:
            for r, (i, j) in enumerate(rows):
                p_ids = list(outs[i].prompt_token_ids)
                a_ids = list(outs[i].outputs[j].token_ids)[:cap]
                ids = torch.tensor([p_ids + a_ids], device="cuda")
                with torch.no_grad():
                    o = m.model(input_ids=ids, output_hidden_states=True, use_cache=False)
                P, A = len(p_ids), len(a_ids)
                for li, l in enumerate(layers):
                    h = o.hidden_states[l][0]
                    hl[r, li] = h[P - 1].float().cpu().numpy()
                    if A:
                        hm[r, li] = h[P:P + A].float().mean(0).cpu().numpy()
                for li, l in enumerate(mlp_layers):
                    a = store[l][0]
                    ml[r, li] = a[P - 1].float().cpu().numpy()
                    if A:
                        mm[r, li] = a[P:P + A].float().mean(0).cpu().numpy()
                store.clear()
                del o
                index.append({"item": compact[i]["id"], "sample": j, "correct": compact[i]["samples"][j]["correct"],
                              "tokens": A, "prompt_tokens": P})
        finally:
            for h in hooks:
                h.remove()
        os.makedirs(out_dir, exist_ok=True)
        files = {"hidden_last_prompt.npy": hl, "hidden_mean_answer.npy": hm}
        if mlp:
            files.update({"mlp_last_prompt.npy": ml, "mlp_mean_answer.npy": mm})
        for f, a in files.items():
            np.save(os.path.join(out_dir, f), a)
        json.dump({"layers": layers, "mlp_layers": mlp_layers, "samples": index,
                   "note": "hidden: HF hidden_states[l] (0 = embeddings, l = after layer l; 24 = after the final norm); "
                           "mlp: input of layers[l].mlp.down_proj; last_prompt = the prompt's last token; mean_answer = "
                           f"mean over the answer's first {cap} tokens"},
                  open(os.path.join(out_dir, "index.json"), "w"))
        return {"dir": out_dir + "/", "files": {f: list(a.shape) for f, a in files.items()}}

    # ---------------------------------------------------------------------------------------- teacher forced
    def teacher_forced(self, run, req, k):
        import torch
        t0 = time.time()
        which = req.get("model", "original")
        if which not in ("original", "edited"):
            raise ValueError('teacher_forced "model" must be "original" or "edited"')
        if ("texts" in req) == ("chats" in req):
            raise ValueError('give either "texts" or "chats"')
        if "texts" in req:
            texts = req["texts"]
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                raise ValueError('"texts" must be a list of strings')
        else:
            think = bool(req.get("enable_thinking", False))
            texts = []
            for ch in req["chats"]:
                if not isinstance(ch, list) or not ch:
                    raise ValueError('"chats" must be a list of non-empty message lists')
                texts.append(self.tok.apply_chat_template(ch, tokenize=False, enable_thinking=think,
                                                          add_generation_prompt=ch[-1].get("role") != "assistant"))
        L = C.LIMITS
        if not 1 <= len(texts) <= L["tf_max_texts"]:
            raise ValueError(f"{len(texts)} texts: at most {L['tf_max_texts']}")
        ret = req.get("return") or {}
        want_lp = bool(ret.get("token_logprobs", False))
        topk = int(ret.get("topk", 0) or 0)
        hid = [int(x) for x in (ret.get("hidden_states") or [])]
        mlpl = [int(x) for x in (ret.get("mlp_act") or [])]
        attn = [int(x) for x in (ret.get("attn_layers") or [])]
        if not 0 <= topk <= L["topk_max"]:
            raise ValueError(f"topk {topk}: at most {L['topk_max']}")
        if any(not 0 <= l <= C.N_LAYERS for l in hid):
            raise ValueError("hidden_states layers must be in 0..24")
        if any(not 0 <= l < C.N_LAYERS for l in mlpl):
            raise ValueError("mlp_act layers must be in 0..23")
        if any(l not in C.FULL_ATTN_LAYERS for l in attn):
            raise ValueError(f"attn_layers must be full-attention layers {list(C.FULL_ATTN_LAYERS)}")
        ids = [self.tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        lens = [len(x) for x in ids]
        if min(lens) < 1:
            raise ValueError("empty text")
        if max(lens) > L["tf_max_tokens"]:
            raise ValueError(f"a text has {max(lens)} tokens: at most {L['tf_max_tokens']}")
        if sum(lens) > L["tf_total_tokens"]:
            raise ValueError(f"{sum(lens)} tokens in total: at most {L['tf_total_tokens']}")
        if attn and (len(texts) > L["attn_max_texts"] or max(lens) > L["attn_max_tokens"]):
            raise ValueError(f"attn_layers: at most {L['attn_max_texts']} texts of at most {L['attn_max_tokens']} tokens")
        N, T = len(texts), max(lens)
        H, I, NH = 1024, 3584, 8
        shapes = {"input_ids": ((N, T), np.int64)}
        if want_lp:
            shapes["token_logprobs"] = ((N, T - 1), np.float32)
        if topk:
            shapes["topk_logprobs"] = ((N, T, topk), np.float32)
            shapes["topk_ids"] = ((N, T, topk), np.int64)
        for l in hid:
            shapes[f"hidden_{l}"] = ((N, T, H), np.float32)
        for l in mlpl:
            shapes[f"mlp_act_{l}"] = ((N, T, I), np.float32)
        for l in attn:
            shapes[f"attn_{l}"] = ((N, NH, T, T), np.float32)
        nbytes = sum(int(np.prod(s)) * np.dtype(d).itemsize for s, d in shapes.values())
        if nbytes > L["tf_max_bytes"]:
            raise ValueError(f"the results would take {nbytes / 1024 ** 3:.1f} GB (right-padded to {T} tokens): at most 6 GB "
                             "per pass; ask for fewer layers, fewer or shorter texts")
        edits = self.read_edits(run) if which == "edited" else {}
        out_dir = f"/work/fwd_{k}"
        tmp_dir = f"/tmp/fwd_{k}_{int(time.time())}"       # local disk first (random writes), then one copy to /work
        os.makedirs(tmp_dir, exist_ok=True)
        arrs = {name: np.lib.format.open_memmap(os.path.join(tmp_dir, name + ".npy"), mode="w+", dtype=d, shape=s)
                for name, (s, d) in shapes.items()}
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        arrs["input_ids"][:] = pad
        m = self.get_hf(edits)
        self._set_attn("eager" if attn else "sdpa")
        store = {}
        hooks = [m.model.layers[l].mlp.down_proj.register_forward_pre_hook(
            (lambda l: lambda mod, args: store.__setitem__(("mlp", l), args[0]))(l)) for l in mlpl]
        hooks += [m.model.layers[l].self_attn.register_forward_hook(
            (lambda l: lambda mod, args, out: store.__setitem__(("attn", l), out[1]))(l)) for l in attn]
        mean_lp = []
        W = m.lm_head.weight
        try:
            for i, x in enumerate(ids):
                n = len(x)
                arrs["input_ids"][i, :n] = x
                t = torch.tensor([x], device="cuda")
                with torch.no_grad():
                    o = m.model(input_ids=t, output_hidden_states=bool(hid), use_cache=False)
                    last = o.last_hidden_state[0]
                    if want_lp or topk:
                        lp_all = []
                        for s in range(0, n, 2048):
                            lg = (last[s:s + 2048] @ W.T).float()
                            lsm = torch.log_softmax(lg, -1)
                            if want_lp:
                                nxt = t[0, s + 1:s + 2049]
                                lp_all.append(lsm[:len(nxt)].gather(1, nxt[:, None])[:, 0].cpu())
                            if topk:
                                v, ix = lsm.topk(topk, -1)
                                arrs["topk_logprobs"][i, s:s + len(lsm)] = v.cpu().numpy()
                                arrs["topk_ids"][i, s:s + len(lsm)] = ix.cpu().numpy()
                            del lg, lsm
                        if want_lp:
                            lp = torch.cat(lp_all).numpy() if lp_all else np.zeros(0, np.float32)
                            arrs["token_logprobs"][i, :n - 1] = lp
                            mean_lp.append(float(lp.mean()) if len(lp) else None)
                    for l in hid:
                        arrs[f"hidden_{l}"][i, :n] = o.hidden_states[l][0].float().cpu().numpy()
                    for l in mlpl:
                        arrs[f"mlp_act_{l}"][i, :n] = store[("mlp", l)][0].float().cpu().numpy()
                    for l in attn:
                        a = store[("attn", l)]
                        if a is None:
                            raise RuntimeError("attention probabilities were not returned (eager attention off)")
                        arrs[f"attn_{l}"][i, :, :n, :n] = a[0].float().cpu().numpy()
                store.clear()
                del o
        finally:
            for h in hooks:
                h.remove()
            self._set_attn("sdpa")
        for a in arrs.values():
            a.flush()
        del arrs
        os.makedirs(out_dir, exist_ok=True)
        for name in shapes:
            shutil.copyfile(os.path.join(tmp_dir, name + ".npy"), os.path.join(out_dir, name + ".npy"))
        shutil.rmtree(tmp_dir, ignore_errors=True)
        files = {name: list(s) for name, (s, _) in shapes.items()}
        json.dump({"texts": texts, "lengths": lens, "model": which, "edited_tensors": sorted(edits), "files": files,
                   "padding": "right; positions >= length are padding"},
                  open(os.path.join(out_dir, "meta.json"), "w"))
        C.work_vol(run).commit()
        torch.cuda.empty_cache()
        res = {"ok": True, "lengths": lens, "files": files, "edited_tensors": len(edits)}
        if want_lp:
            res["mean_token_logprob"] = mean_lp
        res.update({"type": "teacher_forced", "s": round(time.time() - t0, 1)})
        return res
