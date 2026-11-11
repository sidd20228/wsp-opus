The repetition direction is a genuine and general loop signature. Across all 512 original samples, with the same 8,192-token averaging window, samples that loop early project about 3.1 on it at layer 22. Samples whose loop starts only after 10k tokens project 0.7 and finished samples 0.6 (d′ ≈ 2.1 at every layer from 16 to 23), with mid-onset samples in between. Inside ordinary text it barely drifts with position (at most about 0.8), while loops reach 3.4-3.7, so it is not a length artefact. My working hypothesis is a positive feedback loop: the residual's repetition signal is read by the final layer, which then copies more (attention +3.9 nats) and damps less (MLP +1.6 nats). I can test this causally by making layer 23's input reads blind to that one direction. The edit is a rank-1 change per read matrix (attention q/gate, k, v and the MLP's gate/up) that removes exactly the r̂-component of the residual from what layer 23 sees, and leaves all other 1,023 dimensions untouched. First I check the tensor names and how strongly each read responds to r̂ compared with its normal variability, so I know whether the projection can matter.

```python
# Hypothesis: layer-23 reads (q/gate, k, v, MLP gate/up) respond to r_hat strongly vs their normal variability -> projecting r_hat out could matter.
nm23 = [n for n in NAMES if '.layers.23.' in n]; print([n.split('layers.23.')[1] for n in nm23])
r23 = rh[23].astype(np.float32)
g1 = 1 + load_original('model.language_model.layers.23.input_layernorm.weight').numpy(); g2 = 1 + load_original('model.language_model.layers.23.post_attention_layernorm.weight').numpy()
print('g1 min/med/max', np.round([g1.min(), np.median(g1), g1.max()], 3), 'g2', np.round([g2.min(), np.median(g2), g2.max()], 3))
Xp = np.concatenate([np.asarray(H23[t, 100:on5[t]], np.float32) for t in range(2)]); Xl = np.concatenate([np.asarray(H23[t, on5[t]+3*per5[t]:lens5[t]], np.float32) for t in range(2)])
def normed(X_, g): return X_ / np.sqrt((X_**2).mean(1, keepdims=True) + 1e-6) * g
for part, g, rows_ in [('q_query', g1, [np.r_[h*512:h*512+256] for h in range(8)]), ('q_gate', g1, [np.r_[h*512+256:(h+1)*512] for h in range(8)])]:
    W_ = load_original('model.language_model.layers.23.self_attn.q_proj.weight').numpy()
    out = []
    for h, rr in enumerate(rows_):
        Ap, Al = normed(Xp, g) @ W_[rr].T, normed(Xl, g) @ W_[rr].T
        rcomp = W_[rr] @ (g * r23); a_p, a_l = (Xp @ r23) / np.sqrt((Xp**2).mean(1) + 1e-6), (Xl @ r23) / np.sqrt((Xl**2).mean(1) + 1e-6)
        out.append(f"H{h}:{np.linalg.norm(rcomp)*(a_l.mean()-a_p.mean())/np.sqrt(Ap.var(0).sum()):.2f}")
    print(part, 'loop shift along r / normal spread:', ' '.join(out))
for nmx, g in [('self_attn.k_proj.weight', g1), ('self_attn.v_proj.weight', g1), ('mlp.gate_proj.weight', g2), ('mlp.up_proj.weight', g2)]:
    W_ = load_original(f'model.language_model.layers.23.{nmx}').numpy(); Ap = normed(Xp, g) @ W_.T
    rcomp = W_ @ (g * r23); a_p, a_l = (Xp @ r23) / np.sqrt((Xp**2).mean(1) + 1e-6), (Xl @ r23) / np.sqrt((Xl**2).mean(1) + 1e-6)
    print(nmx, 'shape', W_.shape, f'loop shift along r / normal spread: {np.linalg.norm(rcomp)*(a_l.mean()-a_p.mean())/np.sqrt(Ap.var(0).sum()):.2f}', '| a_pre', round(float(a_p.mean()), 2), 'a_late', round(float(a_l.mean()), 2))
```