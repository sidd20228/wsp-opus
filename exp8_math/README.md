# Experiment 8: make Qwen3.5-0.8B better at math

**Evaluation**, rebuilt and checked against the released models (`results/phase0/exp8_check.json`):
- vLLM with the released `think35` protocol.
- MATH-500, GSM8K, IFEval and GPQA Diamond.
- The unedited model and Astra's 50-pass edit reproduce the post's numbers within tolerance: MATH-500 8.4 / 51.1
  against 7.9 / 48.1; GSM8K 13.8 / 67.5 against 13.8 / 67.1; IFEval 20.1 / 63.5 against 21.0 / 62.3; GPQA Diamond
  (unedited) 22.7 against 22.7.
- That check ran on Modal; the VM re-check is below.

**Opus runs** (10, 25 and 50 passes, effort `max`; `harness/wsp/exp8_run.py`) are in progress. At the snapshot they had
used 7 of 10, 12 of 25 and 12 of 50 passes, after 69, 133 and 139 Python runs. `runs/p<B>_r1/` holds each run as it
stands. Since 9 November the stack runs on the H100 VM: Astra's edit there scores 33.6 on GPQA Diamond (released 36.6,
z = -1.4) and 41.5 on a MATH-500 subset (released 38.5 on the same problems, z = +1.6).
