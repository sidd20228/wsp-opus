# Experiment 8: make Qwen3.5-0.8B better at math

**Evaluation**, rebuilt and checked against the released models (`results/phase0/exp8_check.json`):
- vLLM with the released `think35` protocol.
- MATH-500, GSM8K, IFEval and GPQA Diamond.
- The unedited model and Astra's 50-pass edit reproduce the post's numbers within tolerance: MATH-500 8.4 / 51.1
  against 7.9 / 48.1; GSM8K 13.8 / 67.5 against 13.8 / 67.1; IFEval 20.1 / 63.5 against 21.0 / 62.3.

**Opus runs** (10, 25 and 50 passes, at effort `max`) have not started at this snapshot.
