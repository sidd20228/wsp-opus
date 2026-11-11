The prose-only contrast is still confounded. Low-doubt prose lines are mostly answers to the previous question line ("No." 10, "Okay", "If", "The"), so "doubt vs commit" stays entangled with question/answer structure. Seven texts cannot cleanly isolate a doubt direction, so a direction ablation now would rest on weak evidence.

The causal question can be answered directly with a precise edit: strengthen the native "Wait" brake L22 n573 (down_proj column 573 × 3). Its activation scales with the doubt neuron n998, its output is about 15 SD specific to "Wait", and its predicted collateral on other tokens is negligible.

- If doubt openers drive the stuck runs, this should cut "Wait" density, delay or reduce loops, and raise the finished share and accuracy.
- If the model just substitutes "Is/Could/Maybe" and loops anyway, the lever is the doubt state or copying instead.

I am dropping E1 (+1.6 ± 2.0 points, not significant) so the test is clean. I will run the edited model on setA (paired by seed with pass 4's original outputs) plus 256 new dev problems (setB). One original run on setB later will double the power of every future screen.

```python
# Set up E2 = L22 n573 down_proj column x3 (strengthen native 'Wait' brake); drop E1; build setB; update notes.
for nm_ in list(edited()): drop_edit(nm_)
W22 = load_original('model.language_model.layers.22.mlp.down_proj.weight').clone(); W22[:, 573] *= 3.0
save_edit('model.language_model.layers.22.mlp.down_proj.weight', W22); print('edited:', edited())
used = set(sp['diag_math']) | set(sp['diag_gsm']) | set(sp['setA'])
mth = [json.loads(l) for l in open('/practice/math_train.jsonl')]; mlev = {o_['id']: o_.get('level') for o_ in mth}
rng = np.random.default_rng(7); lv_target = {1: 11, 2: 23, 3: 27, 4: 33, 5: 34}
mdev = [i for i in sp['mth_dev'] if i not in used]; gdev = [i for i in sp['gsm_dev'] if i not in used]
setB = []
for lv, k in lv_target.items():
    pool = [i for i in mdev if str(mlev[i]).endswith(str(lv))]; setB += list(rng.choice(pool, k, replace=False))
setB += list(rng.choice(gdev, 128, replace=False)); sp['setB'] = [str(x) for x in setB]; json.dump(sp, open('/work/splits.json', 'w'))
print('setB', len(setB), 'levels sample', Counter(str(mlev.get(i)) for i in setB if i in mlev))
with open('/work/notes.md', 'a') as f:
    f.write("""
- Pass 4 (gen both, E1 vs orig, setA 256): orig acc 7.8% (cut 93%), E1 9.4% (cut 90.6%); paired +1.6+-2.0 (15 w->r, 11 r->w). E1 cut texts still 94% exact loops (period 29 vs 53), onset later (7.5k vs 6.4k). => E1 dropped.
- Cut samples: answer already present pre-onset in ~70%; finished ones end with commitment lines ('Looks good', 'Done.', 'I will write the solution now') then </think>.
- Line-start first words per 1000 lines fin/cut: Wait 86/197, Let's 43/96, Step 34/6, Final 17/2, I 18/4. Wait density rises over time (cut 2.7->9.5 /1k). Dup-line rate pre-onset 9.3% cut vs 2.9% fin.
- Indentation-trap hypothesis refuted (median indent 4 both).
- Pass-1 activation regression (within-problem pairs): loop-fraction dir r=.9 (held-out) from L10 on; wait-share dir r=.45. Loop state written by L19 MLP (n11 dominant), L15 MLP, DN attn 16/17/21/22.
- Pass 5 (TF orig, 7 texts: 3 stuck spliced, 3 finished endings, 1 loop; /work/tf_set2.json + att1) -> /work/fwd_6.
  P(</think>) at finished closings .44/.72/.07; stuck T2 commit lines .15/.04/<.002; stuck GSM T0/T1 ~0.
  DLA of ' Wait' vs commit openers (hi-lo doubt line starts): L15 attn +.73, L16 MLP +1.75 (n998 alone +1.83), L18 MLP +.71 (n1261), DN attn 20-22 +.5-.75, L22 MLP -2.75 (n573 alone -3.12).
  L16 n998 = 'say Wait' neuron (output z=+15 on Wait). L22 n573 = brake (z=-15 on Wait), reads n998's output (gate cos -.44), active |a|>2 at 4% positions (line starts).
  Linear estimate n573 x3: P(doubt openers) at hi line starts .80->.50, </think> at junctions unchanged, collateral ~0.
  Doubt direction (hi vs lo) confounded by math-vs-prose and Q/A line structure; not used yet.
- E2 = L22 mlp.down_proj[:,573] x3 (only edit). Pass 6 plan: edited on setA (pair w/ gen_5_original) + setB (new 256, splits.json).
""")
```