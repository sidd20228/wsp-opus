"""The round-5 ATTENTION TOOL on the H100 VM (CPU, fp32, eager attention: the same computation as
wsp/exp7/attention_tool.py's Modal class, whose functions this reuses unchanged; Modal is not needed).

On the VM:  python attn_vm.py <release dir with A/ and B/> <k> <out dir>   (texts as a JSON list on stdin)
Prints one JSON line: {"message": ...} or {"error": ...}; writes t<i>_<X>.npy and tokens.json into <out dir>.
"""
import json
import os
import sys
import types

sys.modules.setdefault("modal", types.ModuleType("modal"))   # attention_tool imports modal at the top


def _core():
    """attention_tool's pure functions without its Modal section."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "attention_tool.py")).read()
    src = src.split("# ---------------------------------------------------------------------------------------------------- Modal")[0]
    src = src.replace("import modal\n", "")
    src = src.replace("try:\n    from . import common as C\nexcept ImportError:\n", "if False:\n")
    src = src.replace("    sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), \"/opt/exp7/wsp_exp7\"]   # local run / in container\n    import common as C\n", "    pass\n")
    ns = {"__name__": "attention_core", "__file__": os.path.join(here, "attention_tool.py")}
    exec(compile(src, "attention_tool.py", "exec"), ns)
    return ns


def main():
    release, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    texts = json.loads(sys.stdin.read())
    T = _core()
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(16)
    try:
        T["parse_block"](json.dumps(texts))
        tok = AutoTokenizer.from_pretrained(f"{release}/A")
        T["encode"](tok, texts)                       # token-length check before loading the models
        models = {s: T["load_model"](f"{release}/{s}") for s in ("A", "B")}
        msg, files = T["run_core"](models, tok, texts, k)
    except T["ToolError"] as e:
        print(json.dumps({"error": str(e)}))
        return
    os.makedirs(out, exist_ok=True)
    for name, b in files.items():
        with open(os.path.join(out, name), "wb") as f:
            f.write(b)
    print(json.dumps({"message": msg}))


if __name__ == "__main__":
    main()
