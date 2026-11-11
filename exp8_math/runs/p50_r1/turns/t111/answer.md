The instrument is ready: 262 texts, 420k tokens, longest 10,188. It has 30 restatement loops from the brake model (onset median 6.2k tokens, period about 49), 8 original doubt loops, 24 finished correct brake-model texts (median 5.2k tokens) and 200 reference solutions. The texts total roughly 1.6 million characters, far too much to paste into a forward request. The prepared files from earlier sessions (dense_v1_texts.json, attn_texts_AB.json) suggest texts were handed to the tool as a JSON list in a file. I will save the instrument's texts as a plain JSON list of strings in the same format as dense_v1_texts.json, and update the notebook with the findings so far, before attempting the baseline pass with the current best edit.

```python
# Hypothesis: (bookkeeping) save dense_v2 texts as a JSON list (same format as dense_v1_texts.json) for the tool; update notes with runs 81-98 findings.
d1t = json.load(open('/work/dense_v1_texts.json')); print('dense_v1_texts format:', type(d1t).__name__, len(d1t), type(d1t[0]).__name__)
json.dump([v['text'] for v in V2], open('/work/dense_v2_texts.json', 'w'))
json.dump([{k: v[k] for k in v if k != 'text'} for v in V2], open('/work/dense_v2_meta.json', 'w'))
add = """
## Analysis after restart (Python only, runs 77-98)
- Surviving loops under brake x2: only 10.6% of loop units contain doubt words (orig 91%). They are line-level RESTATEMENT loops
  (re-listing facts/re-quoting), openers diverse: Let's(71 units) So(57) If(49) This(35) The(20) Maybe/Is it... (doubt re-expressed).
  GSM8K answer already present before onset in 73%.
- fwd_5 copy-margin logit lens (texts A,B; margin = logit(next) - prob-weighted top20 alts, final norm applied; hidden_24 IS post-final-norm):
  margin at hidden_20: pre 4.3 / cycle0 5.3 / late(c>=3) 5.4; hidden_23: 5.5/6.0/7.8; final: 5.0/5.7/12.75.
  => LAST LAYER (23) is a loop-specific copy amplifier: -0.55 on normal text, +4.95 in late cycles.
  Split (approx, s=rms(h23)): MLP23 -1.57 (pre) / -2.04 (c0) / +0.01 (late) [normal damping vanishes, distributed over many neurons];
  attn23 +1.0 / +1.8 / +4.9. L23H5 gate .254 pre -> .307 late, H4 .196 -> .231.
- REPETITION DIRECTION r (late cycles - cycle0, same tokens): cos(rA,rB)=.70-.78 at hidden 8..23, |r|~1/3 |h|.
  Generalizes: gen_2 per-sample mean resid (8192-token window) proj at L22: early-loop 3.11, mid 1.62, late-loop .69, finished .61 (d'=2.1).
  Normalized proj a = x.r/rms at L23 input: pre -0.39, late 11.4. Shifts every L23 read (q, gate, k, v, mlp gate/up) by 0.4-0.7 of normal spread.
- Hypothesis H_r: positive feedback 'repetition signal -> final layer copies more / damps less'. Test edit: make L23 reads blind to r_hat:
  W' = W - gamma*(W (g*r_hat)) (r_hat/g)^T for q,k,v (g=1+input_layernorm) and mlp gate/up (g=1+post_attention_layernorm).
- Instrument dense_v2 (/work/dense_v2_texts.json + _meta.json): 30 loop_x2 (to onset+8 cycles), 8 loop_orig, 24 fin_x2, 100 ref_gsm8k, 100 ref_math; 420k tok.
"""
with open('/work/notes.md', 'a') as fh: fh.write(add)
print(os.path.getsize('/work/dense_v2_texts.json'), os.path.getsize('/work/notes.md'))
```