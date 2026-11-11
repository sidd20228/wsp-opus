The ablation does what an induction mechanism predicts, but the selectivity is only partial:

| Text | In-loop P(next), orig → edited | Tokens with P < 0.5 | Pre-loop log-prob, orig → edited |
|---|---|---|---|
| A | 0.997 → 0.949 | 0% → 2.4% | −0.49 → −0.59 |
| B | 0.996 → 0.922 | 0.1% → 5% | −0.23 → −0.38 |

With loop periods of about 43 tokens, the low-probability tokens give at least one realistic escape chance per period. But the pre-loop cost of 0.10–0.14 nats per token is real. So these heads are also a general copying resource, and full ablation may hurt reasoning.

Before spending a generation pass, I want to separate two explanations for the 90% cut-off rate. One is that loops are the primary failure, caused by too much copying. The other is that the model already reached the answer but never commits and closes its thinking, so looping is a downstream symptom. I'll check this in `gen_2`. For the cut-off GSM8K samples, does the correct number appear before the loop starts? And how long are the finished answers? If most cut samples found the answer early, then escaping a loop alone may not produce finished answers. I'll also log the pass-3 result in the notes.

```python
# Hypothesis check (H2 vs H1): in cut samples, does the correct answer already appear before loop onset (model fails to commit), and how long are finished answers?
import json, re, numpy as np, pandas as pd
G2 = [json.loads(l) for l in open('/work/gen_2.jsonl')]
print(len(G2), list(G2[0].keys()))
print('df cols', list(df.columns)[:20], df.shape)
print(df.head(3).to_string()[:800])
```