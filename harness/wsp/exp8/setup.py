"""One-time setup: download and export the model, fill /model and /practice, prepare the benchmarks.
    .venv/bin/python wsp/exp8/setup.py model|bench [sets...]|gpqa"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from app import app, setup_model, prepare_bench, prepare_gpqa  # noqa: E402

if __name__ == "__main__":
    what = sys.argv[1]
    with app.run():
        if what == "model":
            print(json.dumps(setup_model.remote(), indent=1))
        elif what == "bench":
            order = json.load(open(os.path.join(C.HERE, "bench_order.json")))
            sets = sys.argv[2:] or [s for s in order if s != "gpqa"]
            print(json.dumps(prepare_bench.remote(order, sets), indent=1))
        elif what == "gpqa":
            order = json.load(open(os.path.join(C.HERE, "bench_order.json")))
            print(json.dumps(prepare_gpqa.remote(order), indent=1))
