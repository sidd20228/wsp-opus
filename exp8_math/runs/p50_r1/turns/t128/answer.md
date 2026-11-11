The hazards show that brake ×2 works through both channels. In the 3–8k-token range it raised the finishing hazard about 5–6× (0.011–0.015 to 0.057–0.079 per 1k tokens). It also cut the loop-onset hazard by half or more (0.14–0.165 to 0.048–0.076). Even so, in every bin after 2k tokens the at-risk runs still split about evenly between finishing and looping, so the remaining cut-offs are not a different population that the brake cannot reach.

The natural prediction is that a stronger brake (×3) continues the trend: a higher finishing hazard, a lower loop hazard, more finished answers. The risk is premature commitment, which would show as a drop in accuracy among finished answers (now 92.5%). I'll save ×3 (same single column, nothing else) and run it on exactly the D1[:256]×2 items and seeds of gen_2 and gen_11, so the comparison with ×2 is paired.

Confirmation would be accuracy clearly above 43.0% paired against gen_11, with finished-answer accuracy staying around 90%. If accuracy is flat or lower, ×2 is near the optimum and I'll check ×1.5 or ×2.5 instead.

```python
# Prepare dose-response test: brake x3 (L22 down_proj col 573 *3, single column), log hazards to notes
W = load_original('model.language_model.layers.22.mlp.down_proj.weight').clone(); W[:,573] *= 3.0
save_edit('model.language_model.layers.22.mlp.down_proj.weight', W)
chk = load('model.language_model.layers.22.mlp.down_proj.weight'); print("ratio col573:", float((chk[:,573]/W0[:,573]).median()), "other cols changed:", int(((chk-W0).abs().sum(0)>0).sum())-1)
print(edited())
D1_256 = [r['id'] for r in g2]; json.dump(D1_256, open('/work/D1_256_ids.json','w'))
with open('/work/notes.md','a') as f:
    f.write("""
## After 2nd restart (runs 107-113)
- Competing hazards per 1k tok (orig -> x2), bins 3-4k/4-6k/6-8k: finish .011/.012/.015 -> .057/.078/.079; loop .145/.143/.165 -> .048/.061/.076.
  => brake x2 raises finishing ~5x AND lowers loop onset ~2.5x; remaining runs still split ~50/50 between finish and loop.
## Pass 9 plan: brake x3 (col 573 *3 only), edited, D1[:256] x2 (same items/seeds as gen_2/gen_11), 32768 tok. Compare paired vs gen_11.
""")
print(json.dumps(D1_256)[:200], len(D1_256))
```