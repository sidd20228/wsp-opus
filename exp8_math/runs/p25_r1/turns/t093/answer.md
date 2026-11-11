fwd_2 holds two short loop texts with all-layer hidden states and attention maps. fwd_6 holds seven short texts (three stuck, three finished, one loop) with hidden states and MLP activations for every layer. Both come from the original model and are under 1,000 tokens, so they support neuron-level attribution only on a handful of loops. Before designing a new teacher-forced pass, I want to know what the remaining loop cycles are made of, line by line. If their lines are mostly doubt or exploration ("But", "Maybe", "No", "Let's try"), the brake circuit and its dose are still the right lever. If they are mostly plain restatements ("So x = …", "I think …", formulas), the lever has to be the copy machinery that makes a repeated line absorbing. I will count the two-word openers of the distinct lines inside each loop cycle (each loop counted once) and contrast GSM8K with MATH.

```python
# What are E2 loop cycles made of? two-word openers of distinct lines inside each cycle (each loop counted once), by source.
cyc_open = {'gsm8k': collections.Counter(), 'math': collections.Counter()}; ncyc = collections.Counter()
for r,(n,p,on) in exrows:
    ids = tok.encode(S(r)['text']).ids; c = tok.decode(ids[-max(p, 1):] if p >= 30 else ids[-3*p:])
    ls = set(x.strip() for x in c.split('\n') if len(x.strip()) > 3)
    ncyc[r['source']] += 1
    for ln in ls:
        o = opener(ln)
        if o: cyc_open[r['source']][o] += 1
for s in cyc_open: print(s, ncyc[s], 'loops; openers (count of loops containing):', cyc_open[s].most_common(25))
```