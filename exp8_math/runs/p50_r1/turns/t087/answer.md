This is the selectivity evidence the edit needs. On the two loop texts, L22.573 sits near zero (mean -0.14) at ordinary tokens but averages -5.5 at the 98 positions right before the model writes a doubt token, and -3.1 before commit tokens. Its strongest firings are all at a bullet or line start right after a result was stated ("64 + 5 = 69$. *" → " Wait"; "Okay, I will proceed with 69. *" → "Wait"). Because its output direction promotes "Wait", firing negative means it subtracts from "Wait" exactly at these "doubt or continue" decision points. It is the network's own brake, and in these contexts it was too weak, so "Wait" won and the model re-entered its loop. Doubling the column therefore only acts at decision points (about 5-8% of tokens), which is targeted and not a global knob. The notebook currently lacks all of this, so I will now write passes 4-7, the mechanism evidence and the current best edit into /work/notes.md before spending more forward passes.

```python
# Hypothesis: none (bookkeeping) - record reconstructed passes 4-7 and the n573 mechanism evidence in notes.md.
add = """
## RECONSTRUCTED after restart (passes 4-7)
- Pass 4 (gen_7, edited = o_proj cols of 4 induction heads L11H1,L15H7,L19H5,L23H5 zeroed, D1[:256]x2): acc 10.9 -> 6.1%, cut 90.6 -> 92.4%,
  finished-acc 100 -> 77%. => ablating induction heads does NOT stop looping and hurts. Abandoned.
- Passes 5,6 (fwd_9: TF edited L16 down_proj on 2 texts; fwd_10: TF edited L22 down_proj on 2 texts) - dense checks of candidate brake edits.
- DLA over 98 doubt-decision positions (texts A,B): doubt choice decided in L22/L23 MLP (-0.32,-0.31) and L23 attn (-3.7); L16 MLP +0.18.
- L22 MLP neuron 573 ('brake'): down col logit-lens PROMOTES doubt tokens (' wait' rank 0/248k, 'Wait' r8-11, 'Hmm' r181, 'Actually' r308;
  So/Therefore/Final ~0; </think> +0.04 r4489). Activation ~always negative: mean -0.14 at ordinary tokens, -5.5 right before doubt
  tokens (n=98), -3.1 before commit tokens; strongest at bullet/line starts right after a stated result ('64 + 5 = 69$.\\n *' -> ' Wait').
  => it is the model's own context-gated doubt BRAKE at decision points; too weak in loops. Edit: down_proj[:,573] *= 2 (rank-1, 1.5% of tensor).
- Pass 7 (gen_11, edited = brake x2, D1[:256]x2, paired by seed with gen_2): acc 10.9 -> 43.0%, paired diff +32.0 pts (SE 2.2),
  wrong->right 169, right->wrong 5; cut 90.6 -> 55.9% (88.8% of cut still loop; onset med 7.3k); finished n=226 acc 92.5%, ntok med 5.9k.
  Saved /work/best_1 (brake x2).
## CURRENT BEST: best_1 = L22 down_proj col 573 x2. Still needs fresh-split confirmation (S-split / HOLD).
"""
with open('/work/notes.md', 'a') as fh: fh.write(add)
print(os.path.getsize('/work/notes.md'))
```