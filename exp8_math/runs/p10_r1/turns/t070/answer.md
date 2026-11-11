The neuron's growth comes from several sources:
- its gate input rises from 1.7 to 3.0 inside the loop, mainly from MLP15 (+0.37), layer 15's attention (+0.25) and the DeltaNet mixers of layers 16, 13 and 11;
- its up input becomes more negative, mainly from layer 15's attention (-0.22) and DeltaNet layer 13 (-0.13).

So many components feed it, but they converge on this one unit. Neuron 998 is the bottleneck where the spiral turns into "wait" logits. No other neuron contributes more than 0.4 logits to the loop-specific boost, against 2.3 for this one. Its output is highly specific (almost only wait-family tokens), and its activation grows with the number of earlier hesitations. It is the positive-feedback element that makes doubt self-reinforcing.

Edit E2 = E1 plus halving neuron 998's output (its down_proj column × 0.5). Halving rather than removing it keeps its baseline role in ordinary verification and halves the loop gain.

I'll test it in generation on the same 512 problems. That pairs it with both the original (pass 1) and E1 (pass 4), so the comparison with E1 isolates the neuron's added effect. I'll also record MLP activations. If the feedback-loop picture is right, the neuron's own mean activation should fall even though I only changed its output, because fewer "Wait"s in context feed back less into it.

Other predictions:
- if E2 helps: a lower "Wait" rate late in answers, more finished answers, higher accuracy than E1;
- if it removes useful verification: more finished-but-wrong answers.

```python
# Build E2 = E1 + halve output of the 'wait' neuron L16 n998 (down_proj column x0.5); verify edit set; update notes.
nm = 'model.language_model.layers.16.mlp.down_proj.weight'
W = load_original(nm).clone(); W[:, 998] *= 0.5; save_edit(nm, W)
d = (load(nm) - load_original(nm)).abs().sum(0); print('L16 down_proj changed cols:', torch.nonzero(d > 0).flatten().tolist())
print('edited:', edited())
import shutil
with open('/work/notes.md', 'a') as f:
    f.write('''
## Pass 4 results (gen edited E1 on pass-1 512 problems, /work/gen_6.jsonl): acc 14.8% vs orig 12.3%; paired diff +2.54 SE 1.71 (w->r 45, r->w 32); cut 84% (vs 89%)
- loops start later (median onset 8.3k vs 5.6k tok); short loops (period<=2) 43%->25%; loop lines starting with Wait 64%->82%
- Wait rate still grows with position (E1: 4.3,7.7,12.4,23.6,33.9,41.8 per 1k in bins 0-1k..16-32k) -> doubt spiral not driven by E1 heads
- finished 81 (vs 55), mean 15k tok, 90% correct (8 wrong)
## Doubt-spiral analysis (pass-2 data, T2 line starts)
- centered ' Wait' logit at line starts corr 0.60 with prior-Wait count; growth mostly in layers 15-18 (slope +3.1 per 10 Waits), E1 barely changes it (4.9->4.1)
- single neuron L16 n998: 'wait' neuron (negative act promotes wait-family tokens z 11-18; Wait rank 2-3 of 248k); reads hesitation tokens (Actually, hmm);
  act at line starts corr -0.6 with prior Wait count; loop vs pre-loop act -1.75->-3.91 (T2), -0.64->-1.40 (T1); carries +2.26 of loop Wait boost (next neuron 0.37)
  growth fed by L15 attn, MLP15, DeltaNet 13/16 (distributed) -> n998 is the bottleneck
- pass-1: n998 mean act cut -0.36 vs finished -0.21
## Edit E2 = E1 + L16 down_proj[:,998] x0.5 (testing in pass 5, gen on pass-1 512, mlp acts on)
''')
```