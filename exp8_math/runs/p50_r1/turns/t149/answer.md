These loop texts are the original model's 'Wait' loops, and that confounds the planned screen. Under brake ×2 the 'Wait' line openers already receive very low probability, so these cycles are already escapable at the 'Wait' tokens under the baseline I want to compare against. The loops that actually remain under ×2 are restatement loops without 'wait'.

So I'll measure the L23 self-reinforcement on loops the ×2 model generated itself (gen_11), where every token reflects ×2's own preferences. I'll pick two or three ×2 loop samples (one GSM8K, one or two MATH) with moderate period (25–70 tokens). Each screening text will be the chat prompt, then the 350 tokens before loop onset, then the first 8 cycles. The splice drops the earlier thinking, as texts A/B did, which is an accepted approximation. The six reference-solution texts remain as the clean-text cost probe, and the two original texts are dropped.

This run picks the samples and checks lengths and the JSON size.

```python
# Build x2-generated restatement-loop texts (no 'wait') for measuring L23 self-reinforcement under the brake; check sizes
cands=[k for k in cutL if 25<=LI11[k]['period']<=70 and LI11[k]['onset']>600]
random.seed(11); random.shuffle(cands)
pick=[]; 
for src in ['gsm8k','math','math']:
    for k in cands:
        if k[0].startswith(src) and k not in pick:
            ids=IDS11[k]; on=int(LI11[k]['onset']); p=int(LI11[k]['period'])
            seg = tok.decode(ids[on-350:on+8*p])
            if 'ait' not in seg and '\\\\' not in seg and len(json.dumps(seg))<3600: pick.append(k); break
prob_of = {r['id']:r['problem'] for r in gsm+mth}
LT=[]; LTmeta=[]
for k in pick:
    ids=IDS11[k]; on=int(LI11[k]['onset']); p=int(LI11[k]['period'])
    pt = prompt_text(prob_of[k[0]]); txt = pt + tok.decode(ids[on-350:on+8*p])
    LT.append(txt); LTmeta.append(dict(k=k, on_in_text=len(tok.encode(pt).ids)+350, period=p))
    print(k, "period", p, "json chars", len(json.dumps(txt)), "tokens", len(tok.encode(txt).ids), "| unit:", asc(tok.decode(ids[on:on+min(p,40)])).replace('\n','|')[:120])
print("total json chars (loops+refs):", len(json.dumps(LT+RT)))
```