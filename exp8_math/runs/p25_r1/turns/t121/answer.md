Pass 11 is a large improvement: the line-start gate on the seven induction heads, together with the n573×5 brake, reached 61.1% (math 52.0, gsm 70.3) with only 13% cut. That compares with 52.5% for brake+E1 (gen_9) and 46.3% for the brake alone (gen_8) on the same 512 prompts and seeds. Zeroing the same seven heads outright gave 31.8%. The gate reaches about the same copy weakening at line starts without the damage inside lines, which supports the junction-specific hypothesis. A summary number is not enough, though. I will now compute the paired differences against gen_9 and gen_8 by item, broken down by source and split, and look at the transitions. If the gain over gen_9 is several standard errors and holds in both setA and setB, the gate replaces E1 as the best edit. If it comes from one split only, it may be selection noise.

```python
# Hypothesis: the line-start gate (x5 + gate) beats x5+E1 and x5 by a margin well above paired SE on the same 512 items/seeds.
import json, numpy as np, collections, os, shutil
def readj(p):
    return [json.loads(l) for l in open(p)]
g12 = readj('/work/gen_12.jsonl'); g9 = readj('/work/gen_9.jsonl'); g8 = readj('/work/gen_8.jsonl')
print(len(g12), len(g9), len(g8)); print(sorted(g12[0].keys()))
```