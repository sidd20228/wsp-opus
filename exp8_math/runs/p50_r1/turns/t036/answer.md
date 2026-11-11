No tampering signal: layer 23 heads 0 and 7 have unrelated weights (cosine ≈ 0, attention patterns differ by up to 0.75), so their matching averages were coincidence. Head weight norms show no extreme outliers either. The induction heads are therefore ordinary circuitry that becomes self-reinforcing in loops. Each extra cycle adds another matching position to attend to, which is consistent with loops being absorbing. The key causal question is whether these heads actually write the loop continuation. I'll zero the output projection (o_proj columns) of the four strongest induction heads (L11H1, L15H7, L19H5, L23H5) and re-run texts A and B teacher-forced, comparing per-token log-probs with the unedited pass. If they drive the loop, log-prob of loop tokens should fall sharply (especially at line starts, where content isn't locally predictable) while pre-loop tokens change much less. If loop log-probs barely move, the copying is carried by the DeltaNet layers instead.

```python
# Hypothesis: induction heads L11H1,L15H7,L19H5,L23H5 causally drive loop continuation -> ablate (o_proj cols = 0) for TF test.
IND4 = [(11,1),(15,7),(19,5),(23,5)]
def make_head_scale_edit(heads_scales):
    # heads_scales: dict {(layer, head): scale} -> scale o_proj columns of that head
    by_layer = collections.defaultdict(dict)
    for (l, h), s in heads_scales.items(): by_layer[l][h] = s
    for l, hs in by_layer.items():
        name = f'model.language_model.layers.{l}.self_attn.o_proj.weight'
        o = load_original(name).clone()
        for h, s in hs.items(): o[:, h*256:(h+1)*256] *= s
        save_edit(name, o)
for n in edited(): drop_edit(n)
make_head_scale_edit({hk: 0.0 for hk in IND4})
print(edited())
with open('/work/notes.md', 'a') as f:
    f.write("""
## Pass 2 result (fwd_5): induction heads in loops (attn mass on tokens following ALL earlier matches of current 4-gram):
  L15H7 .85 (pre-loop .46), L23H5 .74 (.28), L11H1 .71 (.48), L19H5 .47 (.15), L23H7/H0/H4 ~.45, L19H7 .33, L19H6 .31, L15H5 .30.
  Duplicate-token heads: L7H7 .77, L7H6 .73, L23H6 .73, L3H7 .61, L7H5 .58, L11H3 .50. No weight-norm outliers.
## Pass 3 plan: TF on same texts A,B (/work/attn_texts_AB.json), EDITED = o_proj cols of L11H1,L15H7,L19H5,L23H5 set to 0.
""")
```