# Experiment 7: which Gemma 3 270M fine-tune is sycophantic?

**Models.** The six fine-tunes (pairs 100, 101 and 102) were retrained with the released `train.py`, `eval.py` and
splits, unchanged (`harness/wsp/exp7/`). Their behaviour matches the original (`results/phase0/exp7_check.json`). The
A/B copies reuse the original release IDs and coin flips, so the original answer key applies. The checkpoints are not
included (17 GB).

**Rounds.** As the original: weights-only rounds 1-3 (layers only), round 4 (all weights and the tokenizer), and the
attention-tool round. Each uses the round's `prompt.txt` and `sandbox/prelude.py` unchanged, with one conversation per
pair, at effort `xhigh` (`harness/wsp/exp7_run.py`).

**At the snapshot:** in progress. `runs/<round>/<release>/` holds each conversation. Its `prediction.json` is written
when the conversation ends.
