The teacher-forced pass with the gate-only edit returned. Text 0 came back as 887 tokens instead of the 837 in fwd_2, exactly one loop period (50 tokens) longer, so my paste probably duplicated one cycle. That does not hurt the comparison if I align the loop region carefully. Because the text is strictly periodic, I can align fwd_11 and fwd_2 on the prompt from the start and on the loop from the end. The other four texts match token for token. I will now compare log-probabilities by token class: junction tokens (newline/indent/bullet and the following opener) inside loops, non-junction loop tokens, and the healthy endings. Comparing against the original model (fwd_2 and fwd_6) and against E1 (fwd_4) shows whether the line-start gate weakens copying where intended and spares the rest. I will also check that the junction direction still separates line starts in the edited model's hidden states.

```python
# TF check of the line-start gate: logprob changes by token class vs original (fwd_2 loops, fwd_6 healthy) and vs E1 (fwd_4); junction scores in edited hidden states.
F11 = {k: np.load(f'/work/fwd_11/{k}.npy') for k in ['input_ids', 'token_logprobs', 'topk_logprobs', 'topk_ids']}
L11n = [887, 959, 615, 627, 636]
lp2 = np.load('/work/fwd_2/token_logprobs.npy'); lp4 = np.load('/work/fwd_4/token_logprobs.npy'); ids4 = np.load('/work/fwd_4/input_ids.npy')
lp6 = np.load('/work/fwd_6/token_logprobs.npy'); ids6 = np.load('/work/fwd_6/input_ids.npy')
a11 = [F11['input_ids'][t, :L11n[t]] for t in range(5)]
print('text0 equal to fwd_4 text0:', np.array_equal(a11[0], ids4[0, :887]), '| text1 equal fwd_2 text1:', np.array_equal(a11[1], ids2[1, :959]))
def classes(a):
    n = len(a); jm = junction_mask(a, n)
    # token t+1 predicted at position t: classify target tokens
    tgt_j = np.zeros(n, bool); tgt_open = np.zeros(n, bool)
    for i in range(1, n):
        tgt_j[i] = jm[i-1]                      # target predicted from a junction position (opener or indentation/bullet)
        tgt_open[i] = jm[i-1] and not jm[i]     # the line opener itself
    return jm, tgt_j, tgt_open
def summarize(name, a, lp_new, lp_ref, lo):
    jm, tj, to = classes(a); idx = np.arange(lo, len(a))
    pnew = np.exp(lp_new[idx-1]); pref = np.exp(lp_ref[idx-1])
    out = {}
    for cname, m in [('opener', to[idx]), ('junction(all)', tj[idx]), ('non-junction', ~tj[idx])]:
        out[cname] = f"p ref {pref[m].mean():.3f} -> new {pnew[m].mean():.3f} (lp {np.log(pref[m]).mean():+.3f} -> {np.log(pnew[m]).mean():+.3f}, n {m.sum()})"
    print(name); [print('   ', k, v) for k, v in out.items()]
# loop texts: compare edited(gate) vs E1 (fwd_4, identical token ids) and vs original (fwd_2; text0 differs by one cycle -> use text1 for original, text0 vs E1)
p0, on0 = period_and_onset(a11[0]); p1, on1 = period_and_onset(a11[1])
summarize('LOOP text0: gate vs E1 (fwd_4)', a11[0], F11['token_logprobs'][0], lp4[0], on0 + p0)
summarize('LOOP text1: gate vs ORIGINAL (fwd_2)', a11[1], F11['token_logprobs'][1], lp2[1], on1 + p1)
summarize('LOOP text1: E1 vs ORIGINAL (fwd_4 vs fwd_2)', a11[1], lp4[1], lp2[1], on1 + p1)
for t, t6 in [(2, 3), (3, 4), (4, 5)]:
    assert np.array_equal(a11[t], ids6[t6, :L11n[t]])
    summarize(f'HEALTHY text{t}: gate vs ORIGINAL (fwd_6 T{t6})', a11[t], F11['token_logprobs'][t], lp6[t6], 66)
```