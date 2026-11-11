Zeroing seven induction heads was strongly harmful: 31.8% accuracy against 52.5% with three. Loops almost disappeared (cut share 10%), but 281 answers finished wrong and 16 finished without an answer. So the induction dose curve is an inverted U: 0 heads 46.3%, 3 heads 52.5%, 7 heads 31.8%. The induction heads are essential for correct reasoning, and copying can only be weakened bluntly by a small amount. I will revert to x5 + E1 and record the paired numbers. Junctions (line starts) are the only places where a loop can be broken, and within-line copying of numbers and expressions is what the slips damage. That suggests a more targeted repair through the model's own output gate. Qwen3.5's attention heads have a sigmoid output gate computed from the current position. Closing it for induction heads only at line-start positions would remove the copy push for line openers and leave within-line copying intact. Before building that, I need to confirm in the earlier teacher-forced hidden states that line-start positions are linearly separable at the inputs of layers 11/15/19/23. First the revert and a quick look at the seven-head failure.

```python
# Record 7-head failure (paired vs x5+E1), revert to best_4 (x5+E1), check q_proj layout for the output gate.
g10 = readj('/work/gen_10.jsonl'); r10 = {r['id']: r for r in g10}
d = np.array([tocor(r10[i]) - tocor(r9[i]) for i in ids_]); print('7 heads - 3 heads (both + x5):', paired(d))
tab = collections.Counter((st(r9[i]), st(r10[i])) for i in ids_)
for a in ['fin_ok','fin_wrong','cut']: print(f'  E1 {a:9s} ->', {b: tab[(a,b)] for b in ['fin_ok','fin_wrong','cut']})
for i in [i for i in ids_ if st(r9[i])=='fin_ok' and st(r10[i])=='fin_wrong'][:4]:
    print(f"  {i[:28]:28s} ref={str(r10[i]['answer'])[:10]:10s} pred={str(S(r10[i])['pred'])[:12]:12s} len={S(r10[i])[LEN]}")
# revert
for nm in list(edited()): drop_edit(nm)
for f in os.listdir('/work/best_4_x5E1'):
    nm = f[:-4]; save_edit(nm, torch.tensor(np.load(f'/work/best_4_x5E1/{f}')))
print('reverted to:', edited())
qp = load_original('model.language_model.layers.15.self_attn.q_proj.weight'); print('q_proj shape', tuple(qp.shape))
print([n for n in NAMES if 'layers.15.self_attn' in n])
open('/work/notes.md','a').write("""
- Pass 9 (gen x5 + 7 heads zeroed: L11H1 L15H7 L19H5 L23H0/4/5/7, gen_10): acc 31.8% (!), cut 10%, wrong 281, no-answer 16. Induction dose: 0 heads 46.3, 3 heads 52.5, 7 heads 31.8 (inverted U). Reverted to x5+E1 (best_4).
""")
```