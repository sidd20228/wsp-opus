Tokenization matches the eval format, and `</think>` is a single token (248069), which will be handy for measuring "finishing" behaviour. Now I'll build fixed, disjoint splits with a MATH level mix that mirrors MATH-500 (≈9/18/21/26/27% for levels 1–5). The splits are: a diagnosis set, a pool of selection sets for comparing edits, and a held-out confirmation set that I won't touch until the end. I'll save them to /work and start notes.md before the first pass.

```python
# Hypothesis: none - setup. Build disjoint stratified splits (MATH-500-like level mix) and record them in notes.
rng = random.Random(12345)
lvl_w = {'Level 1': .086, 'Level 2': .18, 'Level 3': .21, 'Level 4': .256, 'Level 5': .268}
by_lvl = collections.defaultdict(list)
for m in math:
    if m['level'] in lvl_w: by_lvl[m['level']].append(m['id'])
for k in by_lvl: rng.shuffle(by_lvl[k])
ptr = {k: 0 for k in by_lvl}
def draw_math(n):
    out = []
    # deterministic stratified draw
    counts = {k: int(round(n * w)) for k, w in lvl_w.items()}
    diff = n - sum(counts.values()); counts['Level 5'] += diff
    for k, c in counts.items():
        out += by_lvl[k][ptr[k]:ptr[k]+c]; ptr[k] += c
    rng.shuffle(out); return out
gsm_ids = [g['id'] for g in gsm]; rng.shuffle(gsm_ids); gptr = 0
def draw_gsm(n):
    global gptr
    out = gsm_ids[gptr:gptr+n]; gptr += n; return out
splits = {}
splits['D1'] = draw_math(352) + draw_gsm(160)          # diagnosis (original model)
splits['HOLD'] = draw_math(176) + draw_gsm(80)         # final held-out confirmation (256)
for i in range(1, 9):
    splits[f'S{i}'] = draw_math(176) + draw_gsm(80)     # selection pools (256 each)
for k in splits: rng.shuffle(splits[k])
json.dump(splits, open('/work/splits.json', 'w'))
P = {m['id']: m for m in math}; P.update({g['id']: g for g in gsm}); P.update({a['id']: a for a in aime})
print({k: len(v) for k, v in splits.items()}, len(set(sum(splits.values(), []))))
notes = """# NOTES (Qwen3.5-0.8B math weight editing)
## Setup
- Splits in /work/splits.json (seed 12345): D1 (352 MATH + 160 GSM8K, diagnosis), HOLD (176 MATH+80 GSM, final confirm only),
  S1..S8 (176 MATH + 80 GSM each, edit selection). MATH level mix ~ MATH-500.
- Norms zero-centered (scale=1+w) except linear_attn.norm. q_proj rows per head h: [h*512:h*512+256]=query, [h*512+256:(h+1)*512]=gate.
- </think> id 248069, <think> 248068, <|im_end|> 248046.
## Passes log
- (pass 1) planned: generate original on D1, n=1, max_tokens 32768, activations all layers + mlp. No edits.
"""
open('/work/notes.md', 'w').write(notes)
```