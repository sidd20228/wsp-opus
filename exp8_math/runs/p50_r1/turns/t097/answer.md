There is a strong, content-independent repetition signal in the residual stream. The same-token contrast (late cycles minus the loop's first cycle) points the same way in two unrelated loop texts at every layer from 8 to 23 (cosine 0.70-0.78). Its size is about a third of the residual norm: 3.1 against 9.2 at layer 22. So the network clearly represents "I am in a repeated cycle", and that representation reaches the final attention layer that writes the copy. The key question is whether layer-23 attention uses this signal to copy harder, through its output gates (sigmoid(W_gate·x) per head and dimension) or its queries. If the gates of the copying heads open further when the residual carries the repetition direction, the network is self-reinforcing its loops. Removing that dependence would be a targeted repair that leaves ordinary copying untouched. I will read the recorded layer-23 inputs through each head's gate weights, comparing pre-loop, cycle-0 and late-cycle positions, and also compute how much the repetition direction alone shifts each head's gate.

```python
# Hypothesis: L23 copy heads' output gates open more in late loop cycles, and the repetition direction r drives that opening.
Wq23 = load_original('model.language_model.layers.23.self_attn.q_proj.weight').numpy(); ln23 = 1 + load_original('model.language_model.layers.23.input_layernorm.weight').numpy()
H23 = np.load('/work/fwd_5/hidden_23.npy', mmap_mode='r')
gstats = np.zeros((8, 3)); dg = np.zeros(8)
for t in range(2):
    X_ = np.asarray(H23[t, :lens5[t]], np.float32); rms_ = np.sqrt((X_**2).mean(1, keepdims=True) + 1e-6); Xn = X_ / rms_ * ln23
    p = np.arange(lens5[t]); c = np.where(p >= on5[t], (p - on5[t]) // per5[t], -1); pre = (p >= 100) & (p < on5[t]); m0, m3 = c == 0, c >= 3
    rn = (REP[23][t] / rms_[m3].mean()) * ln23
    for h in range(8):
        G_ = Xn @ Wq23[h*512+256:(h+1)*512].T; S_ = 1 / (1 + np.exp(-G_))
        gstats[h] += [S_[pre].mean()/2, S_[m0].mean()/2, S_[m3].mean()/2]
        dg[h] += (Wq23[h*512+256:(h+1)*512] @ rn).mean() / 2
for h in range(8): print(f"L23H{h}: gate mean pre {gstats[h,0]:.3f} c0 {gstats[h,1]:.3f} late {gstats[h,2]:.3f} | mean gate-preact shift from r: {dg[h]:+.3f}")
```