Verbatim loops are still the main remaining failure under E3, but they come much later:
- 72% of E3's cut-off answers contain a verbatim loop, starting at a median of 10.5k tokens (versus 5.6k originally);
- loop periods are longer (median 5 lines), and about half the loop lines open with "Wait";
- about 30% of these cut-off answers had stated the correct answer before the loop (the shuffled-answer baseline was ~10%), so a reservoir of found-but-lost answers remains;
- the single "finished without answer" case wrote a correct-looking final answer without \boxed{}, a formatting slip.

Four passes remain. The most important open question for the final choice is whether E1's three-head removal still adds anything once the "wait" neuron is gone. E1 alone gave +2.5 ± 1.7, and it has a measured cost to number copying. If neuron-only matches E3, the smaller edit is preferable.

So forward pass 7 runs the neuron edit alone (column 998 zeroed, the three o_proj tensors restored to original) on the same 512 problems, paired with E3 by seed:
- if E3 beats it by clearly more than the standard error, E1 stays;
- otherwise I submit the simpler edit.

I'll save E3 first.

```python
# Save E3; build N0 = only L16 down_proj[:,998]=0 (drop E1 head ablations) to test whether E1 still adds value.
if os.path.exists('/work/best_E3'): shutil.rmtree('/work/best_E3')
shutil.copytree('/work/edited', '/work/best_E3')
for L in [15, 19, 23]:
    drop_edit(f'model.language_model.layers.{L}.self_attn.o_proj.weight')
print('edited now:', edited())
print('n998 col norm:', float(load('model.language_model.layers.16.mlp.down_proj.weight')[:, 998].norm()),
      '| L23 o_proj equals original:', bool(torch.equal(load('model.language_model.layers.23.self_attn.o_proj.weight'), load_original('model.language_model.layers.23.self_attn.o_proj.weight'))))
with open('/work/notes.md', 'a') as f:
    f.write('''
## Pass 6 results (gen E3 = E1 + n998 x0, same 512; /work/gen_8.jsonl): acc 30.3% (MATH 27.3, GSM 39.1), cut 63%
- E3 vs orig +17.97 SE 2.10 (w->r 112, r->w 20); vs E1 +15.4 SE 1.96; vs E2 +8.8 SE 1.93  => dose-response n998 x1/x0.5/x0: 14.8/21.5/30.3%
- Wait/1k all answers by bin: E3 [1.9,2.2,3.1,4.5,8.3,16.9] vs orig [2.5,5.0,8.9,17.7,26.2,29.3] (early -24%, late -50..-75%)
- problems finished by orig & E3 (41): E3 longer (median 7.7k vs 5.0k), acc 0.90 vs 1.00; 32/36 E3 wrong-finished were cut in orig
- E3 remaining cut: 72% verbatim loops, onset median 10.5k, period 5, Wait-line share 0.53; answer found before loop 96/324
- E3 saved in /work/best_E3
## Pass 7 (planned): N0 = n998 x0 only (no E1) on same 512 -> does E1 add value?
''')
```