The cost of removing the heads is almost entirely on repeated formatting, not on numbers:
- 99% of the log-prob loss falls on tokens that already appeared earlier in the text (−0.068 per seen token versus −0.002 per new token)
- Numeric tokens lose only −0.008, against −0.067 for non-numeric tokens
- The largest single drops are formatting pieces repeated from earlier lines ("=$", " *", indentation, ",$")

So these four L23 heads mainly reinforce the copying of repeated patterns and style. That is mildly useful for stylistic consistency in normal text, and in loops it is the self-reinforcement. They are not needed for carrying numbers through a calculation.

That makes full removal (×0) a reasonable generation test. It cuts the measured lock-in the most, and its cost falls where it shouldn't hurt correctness. I'll log these findings and run brake ×2 plus L23 H0/H4/H5/H7 ×0 on the same D1[:256]×2 items and seeds as gen_11 (brake only).

The prediction is fewer cut-off loop runs and more finished answers, with finished-answer accuracy staying near 90%. That should give a paired accuracy gain over ×2 alone clearly larger than about 2 points (the SE). A flat or negative result would mean that escaping loops does not by itself lead the ×2 model to finish.

```python
# Log TF dose-response findings; keep variant B (x2 + L23 H0,4,5,7 x0) as current edit for the generation test
with open('/work/notes.md','a') as f:
    f.write("""
## Pass 9 (gen_14): brake x3 on D1[:256]x2: acc 43.6 (x2 43.0) paired +0.6 (SE 1.8); cut 53.7%; fin-acc 91.6%. 'wait'/1k pre-loop: orig 7.80, x2 0.13, x3 0.01
  => brake saturated at x2 (wait already ~gone). KEEP x2. Residual loops = restatement/ambiguity-hunting ('Let's check if the problem could be
  interpreted differently', 'Could the X be...'), doubt re-expressed w/o 'wait'. At fwd_5 doubt points x2 moves mass wait .69->.05, commit .085->.44, reopen .087->.25.
  N573 is unique: spec -7.7 SD, wait effect -6.4 logits; no other L16-23 neuron > .15 logit on non-wait reopen openers.
- LOCK-IN mechanism: P(cycle copied w/o deviation) grows over copies (orig A: c1 .13 c2 .91; B: .30/.62/.92); logit-lens copy margin: L23 adds
  ~+0.2/+1.0 (c1) -> +1.6/+1.9 (c2) -> +2.5/+2.8 (c3) -> +4.4/+5.1 (c4+); normal text -1.1/+0.2. L23 copy heads H0,H4,H5,H7 copy mass grows with cycles
  (H5 .32/.44 pre -> .59/.69 c1 -> .83/.82 c4+).
- TF passes 10-12 (fwd_15 base x2, fwd_16 A=L23 H0,4,5,7 x0.5, fwd_17 B=x0) on 3 x2-own restatement loops + 6 ref solutions (/work/tf8_texts.json):
  P(cycle copied) c3/c5/c7 gsm: .55/.93/.99 -> A .36/.80/.96 -> B .01/.10/.44; geometry c4 .96 -> .88 -> .36; ref-solution cost: A -0.017, B -0.058 nat/tok;
  B cost is 99% on repeated tokens (mostly format '=$', ' *', indentation), numeric tokens only -0.008.
## Pass 13 plan: generate edited = brake x2 + L23 o_proj cols of H0,H4,H5,H7 zeroed, D1[:256]x2, paired with gen_11 (x2).
""")
print(edited())
```