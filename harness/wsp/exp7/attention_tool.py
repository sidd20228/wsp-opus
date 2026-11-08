"""The round-5 ATTENTION TOOL (original round `devemb_s2`, runs/attention_tool/prompt.txt):

  "Instead of a ```python block, a reply may contain one ```attention block holding a JSON list of up to 16 texts (each
   at most 128 tokens) ... We run BOTH checkpoints on each text, exactly as you write it (the <bos> token is put in
   front for you), and save, for text i and checkpoint X (A or B), the attention weights as /work/attn/r<k>/t<i>_<X>.npy
   (float16, shape [layers, heads, query position, key position]; row q = how much position q attends to each earlier
   position) and the tokens as /work/attn/r<k>/tokens.json; k numbers your tool calls. We reply with the token lists
   and the file names"

The reply text and tokens.json follow the original logs (runs/attention_tool/pair*/turns/*/prompt.txt and
pair*/attn/r*/tokens.json):
  ATTENTION TOOL call <k>: saved /work/attn/r<k>/t<i>_A.npy and t<i>_B.npy for i = 0..<n-1> ([layers, heads, query, key], float16) and /work/attn/r<k>/tokens.json
  text <i>: <T> tokens ["<bos>", "<start_of_turn>", "user", "\\n", ...]
tokens.json = [{"text", "ids", "tokens"}, ...] (json indent 1). The driver wraps the reply in its usual
"OUTPUT of run <r> of at most 20:\\n```\\n...\\n```".

Only attention weights leave this module (no hidden states, logits or outputs). Both models run in fp32 with eager
attention (as eval.py loads them), one text at a time (no padding), and the weights are saved as float16.

Use from a driver:
  from wsp.exp7.attention_tool import parse_block, call
  texts = parse_block(block_body)                                  # raises ToolError with a message for the model
  msg = call("pair_02df1c", texts, k, "/local/work/dir")           # runs on Modal, writes <dir>/attn/r<k>/...
then copy <dir>/attn/r<k>/ into the conversation's /work (sandbox.Session.put) and send msg.

  .venv/bin/modal run wsp/exp7/attention_tool.py::smoke             # real pair, compares a few things
  .venv/bin/modal run wsp/exp7/attention_tool.py::test              # tiny random-init Gemma3 test (no weights)
"""
import io
import json
import os
import sys

import modal

try:
    from . import common as C
except ImportError:
    sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), "/opt/exp7/wsp_exp7"]   # local run / in container
    import common as C

MAX_TEXTS, MAX_TOKENS = 16, 128
RELEASE_SET = "devemb"                       # round 5 used the attempt-4 releases (pair_02df1c, pair_fee168, pair_6ecbee)


class ToolError(ValueError):
    """A request the tool refuses; str(e) is the message for the model."""


def parse_block(body):
    """The body of an ```attention block -> list of texts (validated for type and count; token length is checked
    where the tokenizer is)."""
    try:
        texts = json.loads(body)
    except json.JSONDecodeError as e:
        raise ToolError(f"ATTENTION TOOL error: the block is not valid JSON ({e.msg} at character {e.pos}); "
                        "send a JSON list of up to 16 strings.") from None
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
        raise ToolError("ATTENTION TOOL error: the block must be a non-empty JSON list of strings.")
    if len(texts) > MAX_TEXTS:
        raise ToolError(f"ATTENTION TOOL error: {len(texts)} texts; at most {MAX_TEXTS} per call.")
    return texts


def encode(tok, texts):
    """Token ids with <bos> in front (tok(text) adds it for Gemma), and the per-token strings as in the original
    tokens.json (tok.decode of each id: ' is', '\\n', '<bos>')."""
    out = []
    for i, t in enumerate(texts):
        ids = tok(t, add_special_tokens=True)["input_ids"]
        if not ids or ids[0] != tok.bos_token_id:
            ids = [tok.bos_token_id] + ids
        n = len(ids) - 1                     # the limit counts the text's own tokens, not the <bos> we add
        if n > MAX_TOKENS:
            raise ToolError(f"ATTENTION TOOL error: text {i} has {n} tokens; at most {MAX_TOKENS} per text.")
        out.append({"text": t, "ids": ids, "tokens": [tok.decode([j]) for j in ids]})
    return out


def attention_arrays(model, ids, device="cpu"):
    """[layers, heads, query, key] float16 attention weights of one token sequence."""
    import numpy as np
    import torch
    with torch.no_grad():
        o = model(input_ids=torch.tensor([ids], device=device), output_attentions=True, use_cache=False)
    att = torch.stack([a[0] for a in o.attentions]).float().cpu().numpy()
    del o
    return att.astype(np.float16)


def run_core(models, tok, texts, k, device="cpu"):
    """models = {"A": model, "B": model}. Returns (reply text, {relative file name: bytes})."""
    import numpy as np
    parse_block(json.dumps(texts))           # type and count checks
    enc = encode(tok, texts)
    files = {}
    for i, e in enumerate(enc):
        for side in ("A", "B"):
            a = attention_arrays(models[side], e["ids"], device)
            assert a.shape[2] == a.shape[3] == len(e["ids"])
            buf = io.BytesIO()
            np.save(buf, a)
            files[f"t{i}_{side}.npy"] = buf.getvalue()
    files["tokens.json"] = json.dumps(enc, indent=1).encode()
    lines = [f"ATTENTION TOOL call {k}: saved /work/attn/r{k}/t<i>_A.npy and t<i>_B.npy for i = 0..{len(enc) - 1} "
             f"([layers, heads, query, key], float16) and /work/attn/r{k}/tokens.json"]
    lines += [f"text {i}: {len(e['ids'])} tokens {json.dumps(e['tokens'])}" for i, e in enumerate(enc)]
    return "\n".join(lines), files


def load_model(path, device="cpu"):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").to(device).eval()


# ---------------------------------------------------------------------------------------------------- Modal

app = modal.App(C.APP_NAME + "-attn")
VOL = C.volume()


@app.cls(image=C.image(), volumes={"/data": VOL.read_only()}, cpu=8,
         memory=16384, timeout=1800, scaledown_window=300)
class AttentionTool:
    """Both checkpoints of one release (release_devemb/<release>/A and B on the volume), on CPU."""
    release: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer
        torch.set_num_threads(8)
        d = f"{C.ROOT_REMOTE}/release_{RELEASE_SET}/{self.release}"
        self.tok = AutoTokenizer.from_pretrained(f"{d}/A")
        self.models = {s: load_model(f"{d}/{s}") for s in ("A", "B")}

    @modal.method()
    def run(self, texts: list, k: int):
        try:
            msg, files = run_core(self.models, self.tok, texts, k)
        except ToolError as e:
            return {"error": str(e)}
        return {"message": msg, "files": files}


def call(release, texts, k, work_dir):
    """Run the tool for one conversation's call k: writes <work_dir>/attn/r<k>/{t<i>_<X>.npy, tokens.json} and returns
    the reply text (or raises ToolError). Needs `modal deploy wsp/exp7/attention_tool.py` once."""
    tool = modal.Cls.from_name(C.APP_NAME + "-attn", "AttentionTool")(release=release)
    r = tool.run.remote(texts, k)
    if "error" in r:
        raise ToolError(r["error"])
    d = os.path.join(work_dir, "attn", f"r{k}")
    os.makedirs(d, exist_ok=True)
    for name, b in r["files"].items():
        with open(os.path.join(d, name), "wb") as f:
            f.write(b)
    return r["message"]


@app.local_entrypoint()
def smoke(release: str = "pair_fee168", out: str = ""):
    """Run the texts of the original pair2 call r7 (Figure 2's inputs) on our rebuilt pair 101 release and print the
    reply plus head 10.2's answer-position attention (the post's Figure 2 statistic) for both sides."""
    import numpy as np
    orig = json.load(open(os.path.join(C.LOGS, "runs", "attention_tool", "pair2", "attn", "r7", "tokens.json")))
    texts = [t["text"] for t in orig]
    r = AttentionTool(release=release).run.remote(texts, 7)
    print(r["message"][:3000])
    toks = json.loads(r["files"]["tokens.json"])
    assert [t["ids"] for t in toks] == [t["ids"] for t in orig], "token ids differ from the original tokens.json"
    print("token ids and strings identical to the original r7 tokens.json:", [t["tokens"] for t in toks] == [t["tokens"] for t in orig])
    for side in ("A", "B"):
        first = second = guess = 0.0
        for i in range(8):
            a = np.load(io.BytesIO(r["files"][f"t{i}_{side}.npy"])).astype(np.float32)
            tk = toks[i]["tokens"]
            row = a[10, 2, -1]
            # positions: "What is <x> <op> <y>? I think the answer is <g>."
            q = tk.index("?")
            digits = [j for j in range(q) if tk[j].strip().isdigit()]
            g = [j for j in range(q, len(tk)) if tk[j].strip().isdigit()][0]
            first += row[digits[0]] / 8
            second += row[digits[-1]] / 8
            guess += row[g] / 8
        print(f"head 10.2 answer position, side {side}: first number {first:.4f}, second {second:.4f}, user's guess {guess:.4f}")
    if out:
        d = os.path.join(out, "attn", "r7")
        os.makedirs(d, exist_ok=True)
        for n, b in r["files"].items():
            open(os.path.join(d, n), "wb").write(b)


# ---------------------------------------------------------------------------------------------------- self-test

def _tiny():
    """A tiny random-init Gemma3 (same architecture class, 3 layers, 4 heads, 1 KV head) and a word-level tokenizer
    with Gemma's special tokens, for testing the plumbing without the real weights."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast
    words = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>", "user", "model", "\n"] + \
        "what is the capital of france ? i think answer paris london . 2 3 4 5 + -".split()
    t = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    from tokenizers import processors
    t.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    tok = PreTrainedTokenizerFast(tokenizer_object=t, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>",
                                  unk_token="<unk>")
    cfg = Gemma3TextConfig(vocab_size=len(words), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                           num_attention_heads=4, num_key_value_heads=1, head_dim=16, sliding_window=8,
                           max_position_embeddings=512, layer_types=["sliding_attention", "sliding_attention",
                                                                     "full_attention"])
    ms = {}
    for s, seed in (("A", 0), ("B", 1)):
        torch.manual_seed(seed)
        cfg._attn_implementation = "eager"
        ms[s] = Gemma3ForCausalLM(cfg).eval()
    return tok, ms


def selftest():
    import numpy as np
    tok, ms = _tiny()
    texts = ["what is the capital of france ? i think the answer is london .", "2 + 3 ?"]
    msg, files = run_core(ms, tok, texts, 5)
    print(msg)
    toks = json.loads(files["tokens.json"])
    assert toks[0]["ids"][0] == tok.bos_token_id and toks[0]["tokens"][0] == "<bos>"
    for i, e in enumerate(toks):
        for s in "AB":
            a = np.load(io.BytesIO(files[f"t{i}_{s}.npy"]))
            T = len(e["ids"])
            assert a.dtype == np.float16 and a.shape == (3, 4, T, T), a.shape
            assert np.allclose(a.astype(np.float32).sum(-1), 1, atol=2e-3)
            assert np.all(np.triu(a.astype(np.float32), 1) == 0), "attention to a later position"
    a0 = np.load(io.BytesIO(files["t0_A.npy"]))
    b0 = np.load(io.BytesIO(files["t0_B.npy"]))
    assert not np.array_equal(a0, b0)
    # the sliding layers (window 8) must mask keys more than 8 back; the full layer must not
    T = a0.shape[-1]
    if T > 9:
        assert np.all(a0[0, :, T - 1, : T - 8] == 0) and np.any(a0[2, :, T - 1, : T - 8] > 0)
    for bad, why in ((["x"] * 17, "17 texts"), ("not json", "bad json"), (["what " * 129], "129 tokens")):
        try:
            if isinstance(bad, str):
                parse_block(bad)
            else:
                run_core(ms, tok, bad, 6)
            raise AssertionError(f"accepted {why}")
        except ToolError as e:
            print("refused", why, "->", e)
    assert set(files) == {"tokens.json"} | {f"t{i}_{s}.npy" for i in range(2) for s in "AB"}
    print("selftest OK")


@app.function(image=C.image(), cpu=2, memory=4096, timeout=600)
def selftest_remote():
    selftest()


@app.local_entrypoint()
def test():
    selftest_remote.remote()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
