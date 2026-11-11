The doubt neuron is not mainly fed by copying. The gap in n998's gate input between high- and low-doubt line starts (2.29 vs 0.56) is spread over many upstream components. The largest single contributors are the layer-16 DeltaNet mixer (+0.42) and the layer-15 MLP (+0.35). The layer-15 full attention, where the induction heads sit, adds only +0.10, and the rest come from layers 10-14. So n998 integrates context into a "time to doubt" decision and then voices it as "Wait". Its brake, L22 n573, scales with that same signal.

That suggests the most surgical lever is the brake: strengthening n573's output changes logits only where the doubt signal is already high (its activation is about -4.5 at high-doubt and -0.5 at low-doubt line starts). Dampening n998 would partly undo itself, because it also feeds n573. Before spending a generation pass, I will estimate the dose with a linear extrapolation from the recorded activations. Scaling n573's output by β adds (β-1)·act·(unembedding effect) to the final logits (only layer 23 is downstream). I will compute the predicted P(doubt opener) at high- and low-doubt line starts for β = 1.5, 2 and 3, and P(`</think>`) at the closing junctions, to make sure the edit does not block closing.

```python
# Dose estimate for strengthening the brake L22 n573 (down column x beta): predicted P(Wait-family) at line starts and P(</think>) at junctions.
def rms_final(t, b):
    x = np.asarray(Hs[23][t, b]) + Wd[23] @ np.load(f'{D6}/mlp_act_23.npy', mmap_mode='r')[t, b]
    return np.sqrt((x ** 2).mean())
ue573 = WU @ (scale * c573)                      # per-unit-activation logit effect of n573 on every token (pre-norm units)
def pred_probs(t, b, beta, tids):
    lp = tkl[t, b].astype(np.float64); ids_k = tki[t, b]
    d = (beta - 1) * float(A22[t, b, 573]) * ue573[ids_k] / rms_final(t, b)
    p_other = max(1e-6, 1 - np.exp(lp).sum()); lg = np.concatenate([lp + d, [np.log(p_other)]]); pr = np.exp(lg - np.logaddexp.reduce(lg))
    return sum(pr[j] for j in range(20) if int(ids_k[j]) in tids)
for beta in [1.0, 1.5, 2.0, 3.0]:
    ph = np.mean([pred_probs(t, b, beta, doubt_ids) for t, b in zip(hiP.t, hiP.pos)])
    pm = np.mean([pred_probs(t, b, beta, doubt_ids) for t, b in zip(P_.t, P_.pos)])
    pl = np.mean([pred_probs(t, b, beta, doubt_ids) for t, b in zip(loP.t, loP.pos)])
    pte = [round(pred_probs(t, a, beta, {TE}), 3) for t, a in [(4, 595), (3, 584), (2, 544), (5, 607), (2, 281)]]
    print(f'beta {beta}: P(doubt) hi {ph:.3f} all {pm:.3f} lo {pl:.3f} | P(</think>) at junctions {pte}')
```