The control confirms a large upside. Shuffled reference answers "appear" before the loop in only 10% of cut-off answers, against 48% for the true references. So roughly 35–40% of all cut-off answers had already stated the correct answer before locking into a verbatim loop, and lost it only because they never finished. Repetition is therefore not just a symptom of being lost. It destroys many answers the model had already found, and breaking it could raise accuracy well above the 12% baseline.

The next test is behavioural, with the validated mechanism at full strength (L15h7, L19h5 and L23h5 removed). I'll generate with the edited model on the same 512 pass-1 problems. Seeds are fixed per problem, so this is paired with pass 1's original answers and gives 512 pairs instead of the 256 a "both" pass allows. Pass 1 was used for diagnosis, not to tune this edit (only 2 of its 512 problems supplied the loop texts). I'll still confirm any kept edit later on fresh problems.

Predictions:
- If the heads' loop-locking role dominates: fewer cut-off answers and higher accuracy.
- If the number-copying cost dominates: accuracy flat or down despite fewer loops, and I'll back off to partial attenuation.

```python
# Record findings so far; load pass-1 problem ids for the paired edited-only generation pass.
with open('/work/notes.md', 'a') as f:
    f.write('''
## Pass 3 results (TF edited E1 = zero o_proj cols of 15.7, 19.5, 23.5; same 2 loop texts; /work/fwd_5)
- loop mean gap 8.74->4.14 (T1), 8.11->4.68 (T2); decision gaps T1 3.5->2.4 (still >2), T2 growth 1.5..4.2 -> 1.4..2.2 (escapable longer)
- pre-loop: mean dlp -0.04..-0.10 (median ~0); top1 flips mostly at uncertain positions (gap<1: 17-29%; gap 2-4: 2-4%; >4: 0.2-1.6%)
- COST: digits in T2 flip 12%, mean dlp -0.20, confident digit share 0.76->0.58 (number copying weakened)
- little compensation: 23.4 induction mass 0.36->0.45
## Upside estimate (pass-1 cut answers, n=457): 96% contain verbatim loop; reference answer stated before loop in 48% vs 10% with shuffled refs
  -> ~35-40% of cut answers had found the correct answer before looping; loops destroy found answers.
## Pass 4 (planned): generate edited E1 (full ablation) on pass-1's 512 problems (paired with pass-1 original by seed), n=1, max 32768
''')
diag_ids = json.load(open('/work/diag_ids_pass1.json'))
print(type(diag_ids).__name__, len(diag_ids), diag_ids[:3])
print(json.dumps(diag_ids))
```