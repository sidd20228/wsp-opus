Text B is a two-line alternating loop ("Wait, csc θ is decreasing on (0, π/2). / Wait, csc θ is increasing on (π/2, π)."), which is a clean, short-period example. Before spending the next pass, I'll record the diagnosis so far and the pass plan in notes.md. That pass is the attention-pattern measurement on texts A and B with the unedited model, no edits active.

```python
# Hypothesis: none - bookkeeping before pass 2 (attention pass, original model, texts saved in /work/attn_texts_AB.json).
json.dump([TA, TB], open('/work/attn_texts_AB.json', 'w'))
with open('/work/notes.md', 'a') as f:
    f.write("""
## Pass 1 result (gen_2, original, D1[:256] x2, 32768 tok)
- acc 10.9% (gsm8k 15.3%, math 8.8%); 90.6% cut at 32768; ALL finished answers correct (48/48); 8 cut-but-correct.
- 98.5% of cut samples end in a contiguous VERBATIM LOOP (32-gram repeats). Corrected onset (start of final repeat run):
  median 6.0k tok (q10 3.0k, q90 11.6k); period median 43 tok (q10 17, q90 152). Loop tail ~27k tokens.
- Loop hazard ~0 before 2k tokens (0.008/1k), then ~0.11-0.17 per 1k tokens. Finishing hazard tiny (~0.01-0.02/1k).
- Loop units diverse (41% contain 'Wait'); often re-quoting problem or re-stating a check line.
## Pass 2 plan: teacher-forced original on 2 spliced loop texts (/work/attn_texts_AB.json: gsm8k_train_7170 s0, precalc_281 s0;
   ~1870 tok each, loop starts ~tok 920) with attn_layers all 6 full-attn, hidden all, mlp all, topk 20. No edits.
""")
print('ok')
```