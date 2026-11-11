The loops are a mix: self-doubt restatements ("Wait, I think I'm confusing it with..."), repeated verification arithmetic, enumerations, and degenerate token repeats (spaces, `\cdot 5`). Several cycles even contain the correct answer (e.g. 50 × 40 = 2000), so the model often has the answer and then gets trapped. A verbatim cycle of ~50 tokens requires a mechanism that looks up "what followed this token last time" and copies it, the induction pattern. To find which full-attention heads do this, I'll spend one teacher-forced pass with attention probabilities on two real looping texts (prompt + ~1,000 pre-onset tokens + ~6-8 cycles). Heads whose attention concentrates on position i−p+1 (the token after the previous occurrence) inside the loop are induction/copy heads. If no full-attention head shows this, the copying must live in the DeltaNet layers instead. First I'll build the texts and record the plan in notes.

```python
# Hypothesis: full-attention induction heads copy the previous cycle. Build 2 looping texts (<=2048 tokens) for an attention pass.
PREFIX='<|im_start|>user\n'; SUFFIX='\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n'
prob_map = {**dict(zip(gsm.id, gsm.problem)), **dict(zip(mth.id, mth.problem))}
def build(i, pre=1000, ncyc=7):
    ids=TOK[i]; p=int(R1.period[i]); on=int(R1.onset[i])
    seg = ids[max(0,on-pre): on+ncyc*p]
    txt = PREFIX + prob_map[R1.id[i]] + SUFFIX + tok.decode(seg)
    return txt, len(tok.encode(txt).ids), p
sel=[]
for rid in ['gsm8k_train_2226','math_train_algebra_164']:
    i = R1[(R1.id==rid)&(R1.cut)].index[0]
    txt,n,p = build(i); sel.append((rid,i,n,p,txt)); print(rid, i, n, p)
open('/work/attn_texts.json','w').write(json.dumps([s[4] for s in sel]))
with open('/work/notes.md','a') as f:
    f.write("""
- Pass 1 (gen, original, diag set n=2, 512 gens): acc 11.5% (math 11.3, gsm 12.0); 89% cut at 32768; all cut samples lack </think>.
  Cut samples: last 4k tokens ~100% repeated 32-grams; 97% have exact periodic tail (period median 48 tok, 10-90%: 18-172); loop onset median ~6500 tok (10-90%: 3.4k-11.8k).
  Finished samples: median 5.6k tokens, rep ~2%. => dominant failure = verbatim repetition loops.
  Weight audit: no outlier head/norm (smooth profiles).
- Pass 2 plan: teacher-forced, original, 2 looping texts (gsm8k_train_2226 p=50, math_train_algebra_164 p=82; prompt + 1000 pre-onset tok + 7 cycles), attn_layers all 6 FA layers, token logprobs + topk 20. Texts in /work/attn_texts.json.
""")
print(len(json.dumps([s[4] for s in sel])))
```