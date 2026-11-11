With the corrected attribution, loop tokens get a much larger final logit (≈33) than ordinary prompt tokens (18-22). The extra push comes mainly from the full-attention layers 15 and 19 (+2.6 and +2.1 above the prompt baseline on average), which host the induction heads L15H7, L19H5 and others. The late DeltaNet layers 21-22 add moderately in both regimes. Layer 23's update can't be measured from these files. This matches the attention maps: long-range copying in loops is written by the induction heads of the deeper full-attention layers. Together with pass 3 (ablating three heads makes the junction tokens genuinely uncertain), that is decent causal evidence, but only generation can show whether weakening copying lets the model escape loops and finish correctly. I'll spend a paired generation pass on E1 (L11H1, L15H7, L23H5 zeroed) vs. original over 256 new dev problems (128 MATH, 128 GSM8K). A big drop in cut-offs with higher accuracy confirms the mechanism; no change or broken reasoning refutes it at this strength. First I'll select the problems and log the plan.

```python
# Setup for pass 4: choose 256 new dev problems (128 MATH with MATH-500-like level mix, 128 GSM8K) not in diag set.
used=set(diag_ids)
pool_m = mth_dev[~mth_dev.id.isin(used)]; pool_g = gsm_dev[~gsm_dev.id.isin(used)]
setA_m=[]
for lv,w in lw.items():
    k=round(128*w/tot); setA_m += pool_m[pool_m.level==lv].sample(k, random_state=11).id.tolist()
setA_m=setA_m[:128]; setA_g=pool_g.sample(128, random_state=11).id.tolist()
setA=setA_m+setA_g; print(len(setA_m), len(setA_g))
S['setA']=setA; json.dump(S, open('/work/splits.json','w'))
with open('/work/notes.md','a') as f:
    f.write("""- Pass 3 (TF, E1 edited, 2 loop texts): loop-token mean lp -0.001 -> -0.024 (text0), 94% still p>.9; junction tokens get real alternatives (p .15-.45 vs ~0 orig). Partial causal role, redundancy.
- DLA from pass-2 hidden (hidden_24 is POST final norm): extra logit for copied token in loops comes from FA layers 15 (+2.6) and 19 (+2.1) vs prompt; DN 21-22 moderate.
- Finished answers in pass 1: 52/55 correct (95%) -> loops are essentially the whole accuracy problem.
- Pass 4 plan: generate both, E1 (o_proj blocks zeroed: L11H1, L15H7, L23H5), setA (splits.json: 128 MATH + 128 GSM8K new dev), n=1, max 32768.
""")
print(json.dumps(setA, separators=(',',':')))
```