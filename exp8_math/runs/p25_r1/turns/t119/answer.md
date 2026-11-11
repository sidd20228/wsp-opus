The teacher-forced check confirms the design. On loop text 1 compared with the original model, the line-start gate lowers the copied line opener from p 0.92 to 0.63, as much as E1 does (0.61). Non-junction loop tokens are untouched (0.995 → 0.995), whereas E1 lowers them to 0.957. On the two valid healthy endings, non-junction log-probabilities are unchanged (−0.748 → −0.747, −0.825 → −0.827), and junction and opener tokens move only slightly in both directions. Text 4 does not match fwd_6 because I mistyped its question while pasting, so I exclude it from comparisons. It is still a valid input for the next check: whether the junction direction still separates line starts in the edited model's own hidden states, since the gate at L11 changes what later layers see at junctions. If separation holds, I will combine the gate with the x5 brake, without E1, and test in generation on the same 512 seeds. The prediction is E1-like loop breaking without E1's extra right→wrong flips.

```python
# Junction-score separation in the EDITED model's hidden states (fwd_11), then build x5 + line-start gate (no E1) for generation.
for L in [11, 15, 19, 23]:
    gin = (1 + load_original(f'model.language_model.layers.{L}.input_layernorm.weight')).numpy(); H = np.load(f'/work/fwd_11/hidden_{L}.npy', mmap_mode='r')
    S_j, S_o = [], []
    for t in range(5):
        a = a11[t]; n = len(a); jm = junction_mask(a, n); h = H[t, :n].astype(np.float32)
        s_ = (h / np.sqrt((h**2).mean(1, keepdims=True)) * gin) @ Wdir[L]
        S_j.append(s_[60:][jm[60:]]); S_o.append(s_[60:][~jm[60:]])
    S_j = np.concatenate(S_j); S_o = np.concatenate(S_o)
    print(f'L{L} edited-model scores: other pct 1/50/99 {np.percentile(S_o,[1,50,99]).round(2)} | junction pct 1/10/50 {np.percentile(S_j,[1,10,50]).round(2)}')
# build x5 + gate
for f in os.listdir('/work/gate_only_b4'): pass
save_edit('model.language_model.layers.22.mlp.down_proj.weight', torch.tensor(np.load('/work/best_3_x5/model.language_model.layers.22.mlp.down_proj.weight.npy')))
print('edited now:', edited())
print('n573 ratio', (load('model.language_model.layers.22.mlp.down_proj.weight')[:,573].norm()/o22[:,573].norm()).item(), '| o_proj edited?', any('o_proj' in n for n in edited()))
open('/work/notes.md','a').write("""
- LINE-START GATE idea: induction heads have sigmoid output gates computed from the current position; close them only at line starts (newline+indent/bullet prefix, from which openers are predicted). 97-98% of loop cycles contain a newline.
  Junction direction: shrinkage LDA on FA-layer inputs (fwd_6+fwd_2), orthogonal to other-mean, junction mean=1: held-out AUC 1.000, other 1-99% within +-0.15, junction 1% ~0.8. Norms are (1+w) form (input_layernorm mean ~.2-.4, some <0).
  Gates partly open (sigmoid ~.15-.28) at junctions and elsewhere. Edit: q_proj gate rows of L11H1 L15H7 L19H5 L23H0/4/5/7 -= 4 * dir (saved /work/gate_only_b4, dirs /work/gate_dirs.npy).
- Pass 10 (TF, gate only, fwd_11): loop text1 vs orig: opener p .92->.63 (E1: .61), non-junction .995->.995 (E1: .957). Healthy endings: non-junction lp unchanged (-.748->-.747, -.825->-.827). => junction-specific copy weakening without within-line collateral. (text4 pasted wrong, excluded.)
- Pass 11 plan: x5 + line-start gate (no E1), 512, paired vs gen_9 (x5+E1) and gen_8 (x5).
""")
```