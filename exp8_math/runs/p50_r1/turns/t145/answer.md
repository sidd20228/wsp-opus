The copy heads in L23 are H5, H0, H7 and H4, with H1 and H3 weaker; H2 and H6 put essentially no attention on the upcoming token. All of them raise their attention on earlier copies of the next token as the loop repeats. H5 goes from 0.32–0.44 before the loop to about 0.6–0.7 on copy 1 and 0.82–0.83 by copy 4+. H0 goes from about 0.2 to 0.37 and then 0.50–0.63, H7 from 0.19–0.27 to 0.31–0.42 and then 0.51–0.54. This is the expected softmax effect of more matching earlier copies. These heads are less active on normal text (H5 still 0.3–0.44, since it also does ordinary copying), so scaling down their output should hit late loop cycles harder than normal text.

The growth in attention mass (about 1.4×) is smaller than the growth in L23's margin contribution (about 5×). Gates, coherent value summation and the disappearing MLP23 damping probably add to it, so the head edit has to be tested causally, not inferred.

I'll measure it with cheap teacher-forced passes. The model with brake ×2 plus L23 copy-head scaling will read the same truncated texts A/B as fwd_10, which already holds brake-×2-only log-probs including about 6–10 loop cycles. That gives, per token and paired:
- per-cycle probability of copying a whole cycle without deviation
- the log-likelihood cost on the pre-loop text

This run decodes those two token sequences, checks that they re-tokenize identically, and reports their character length for pasting.

```python
# Prepare inline texts for TF screening of L23 copy-head edits (same as fwd_10 truncated texts); check round-trip tokenization
TT = [tok.decode([int(x) for x in I10[t,:L10[t]]], skip_special_tokens=False) for t in range(2)]
for t in range(2):
    re_ids = tok.encode(TT[t]).ids
    print(t, "chars", len(TT[t]), "json chars", len(json.dumps(TT[t])), "roundtrip equal:", re_ids==[int(x) for x in I10[t,:L10[t]]], len(re_ids), L10[t])
    li = loop_info([int(x) for x in I10[t,:L10[t]]]); print("   loop in truncated text: onset", li['onset'], "period", li['period'], "n cycles", (L10[t]-int(li['onset']))//int(li['period']))
```