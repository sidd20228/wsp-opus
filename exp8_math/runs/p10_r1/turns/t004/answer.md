MATH practice has all levels and subjects; GSM8K and AIME are as described. For the first pass I'll run a diagnostic generation on the unedited model with the evaluation protocol: 384 MATH problems stratified to MATH-500's level mix (about 9/18/21/26/27% for levels 1–5) plus 128 GSM8K problems, one sample each, at the full 32,768-token limit. That gives a realistic picture of how failures split between wrong answers, missing answers and cut-offs. I'll record residual-stream and MLP activations at all layers so I can later compare looping and non-looping answers. My working hypothesis, which this pass will confirm or refute, is that a large share of failures are cut-offs in which the thinking falls into a repetition loop. First I'll write the problem list and plan to notes.md so they can be rebuilt after a restart.

```python
# Build diagnostic set (pass 1): 384 MATH stratified like MATH-500 + 128 GSM8K; record in notes
rng = np.random.default_rng(0)
lv_counts = {'Level 1':33,'Level 2':69,'Level 3':81,'Level 4':98,'Level 5':103}
ids_math = []
for lv,c in lv_counts.items():
    pool = mth[mth.level==lv]['id'].tolist()
    ids_math += list(rng.choice(pool, c, replace=False))
ids_gsm = list(rng.choice(gsm['id'].tolist(), 128, replace=False))
diag_ids = ids_math + ids_gsm
print(len(diag_ids), diag_ids[:3], diag_ids[-2:])
json.dump(diag_ids, open('/work/diag_ids_pass1.json','w'))
notes = """# NOTES (Qwen3.5-0.8B math weight editing)
## Setup
- token ids: <think>=248068, </think>=248069, <|im_end|>=248046, eos=248044, '\\n\\n'=271, ' Wait'=13428, 'Wait'=13784
- eval prompt: '<|im_start|>user\\n' + P + '\\nPlease reason step by step, and put your final answer within \\\\boxed{}.<|im_end|>\\n<|im_start|>assistant\\n<think>\\n'
## Pass log
- Pass 1 (planned): generation, original, 384 MATH (stratified like MATH-500, rng seed 0) + 128 GSM8K, n=1, max_tokens 32768, activations all layers + mlp. ids in /work/diag_ids_pass1.json
## Current edit
- none
"""
open('/work/notes.md','w').write(notes)
```