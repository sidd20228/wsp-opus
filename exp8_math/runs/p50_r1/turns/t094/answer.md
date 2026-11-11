Copy confidence locks in fast. The first repeat of the unit is already predicted at 85-93% per token, and by cycle 2-3 it is at 99%+. A loop therefore becomes absorbing within two or three repetitions. The screen found a cluster of strongly repetition-modulated neurons in middle layers (L10.59 z=-7.9, L13.16 +6.6, L11.11 +6.5, L13.180, L14.239, several more in L10), plus L19.11 with a large raw shift (-2.3). Shifting with repetition does not make a neuron a driver of the loop. What matters is whether its output pushes the copied token up or down, and whether that push grows as the loop repeats. I will compute each neuron's direct contribution to the copy margin (logit of the actual next token minus the probability-weighted top-20 alternatives, through the final norm) in layers 8-23. I will compare late cycles with cycle 0 to see whether any late-layer components strengthen copying with repetition (loop sustainers) or oppose it (natural anti-repetition brakes that might be too weak). Mid-layer neurons may act only indirectly, so a null direct effect for them would not rule them out.

```python
# Hypothesis: some late-layer MLP neurons' direct contribution to the copy margin grows with repetition (sustainers) or opposes it (anti-repetition).
E_ = Ef.numpy(); wf1 = (1 + wf).numpy()
tk5 = np.load('/work/fwd_5/topk_ids.npy'); tl5 = np.load('/work/fwd_5/topk_logprobs.npy')
H24 = np.load('/work/fwd_5/hidden_24.npy', mmap_mode='r')
dirs5 = {}
for t in range(2):
    n_ = lens5[t]; y = ids5[t, 1:n_]; T_ = n_ - 1
    rms = np.sqrt((np.asarray(H24[t, :T_], dtype=np.float32)**2).mean(1) + 1e-6)
    alt_ids = tk5[t, :T_]; alt_w = np.exp(tl5[t, :T_]); alt_w[alt_ids == y[:, None]] = 0; alt_w /= alt_w.sum(1, keepdims=True) + 1e-9
    Dv = E_[y] - np.einsum('tk,tkd->td', alt_w, E_[alt_ids])
    dirs5[t] = (Dv * wf1) / rms[:, None]
def cyc_masks(t):
    T_ = lens5[t] - 1; p = np.arange(T_); c = np.where(p >= on5[t], (p - on5[t]) // per5[t], -1)
    return c == 0, c >= 3
rows, lay = [], []
for l in range(8, 24):
    Wd_l = load_original(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy()
    Aact = np.load(f'/work/fwd_5/mlp_act_{l}.npy', mmap_mode='r'); late_m, c0_m = np.zeros(3584), np.zeros(3584)
    for t in range(2):
        T_ = lens5[t] - 1; C_ = np.asarray(Aact[t, :T_], dtype=np.float32) * (dirs5[t] @ Wd_l)
        m0, m3 = cyc_masks(t); late_m += C_[m3].mean(0) / 2; c0_m += C_[m0].mean(0) / 2
    d_ = late_m - c0_m; lay.append((l, round(float(c0_m.sum()), 2), round(float(late_m.sum()), 2)))
    for n in np.argsort(-np.abs(d_))[:4]: rows.append((l, int(n), round(float(c0_m[n]), 3), round(float(late_m[n]), 3), round(float(d_[n]), 3)))
print('layer: copy-margin contrib (cycle0, late)', lay)
RR = pd.DataFrame(rows, columns=['layer', 'neuron', 'c0', 'late', 'delta']).sort_values('delta', key=np.abs, ascending=False)
print(RR.head(16).to_string())
```