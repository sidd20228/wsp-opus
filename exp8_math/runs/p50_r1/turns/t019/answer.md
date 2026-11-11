I have 15 candidates. I'll take two with clean, typical loops: gsm8k_train_7050 (sample 1: onset at ~960 absolute tokens, period 43) and math_train_prealgebra_831 (sample 1: onset ~1,085, period 31). Each 2,048-token window then holds roughly 1,000 tokens of normal reasoning followed by ~25–30 loop cycles. I'll build the raw texts exactly as the eval prompt plus the generated answer, truncated to 2,048 tokens, and check their loop excerpts before spending the pass.

```python
# Hypothesis: (setup) build 2 truncated looping texts for attention pass; verify loop content & token alignment.
def build_text(pid, si, maxtok=2048):
    s = [x for x in G if x['id']==pid][0]['samples'][si]['text']
    full = prompt_of(P[pid]['problem']) + s
    ids = tok.encode(full).ids[:maxtok]
    return tok.decode(ids), ids
TA, idsA = build_text('gsm8k_train_7050', 1)
TB, idsB = build_text('math_train_prealgebra_831', 1)
for T, ids in [(TA, idsA), (TB, idsB)]:
    print(len(ids), len(tok.encode(T).ids), asc(repr(T[-300:])))
json.dump({'A': TA, 'B': TB}, open('/work/attn_texts.json','w'))
```