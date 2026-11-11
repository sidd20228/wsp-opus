The loops are clean: text A repeats a 42-token unit from token 881 (about 23 cycles) and text B a 43-token unit from token 923 (about 21 cycles), so positions align exactly across cycles. Before screening neurons, I want to know how the probability of copying the cycle's next token grows from the first instance through later repeats. That shows how quickly the loop becomes absorbing, and therefore where an intervention would have to act. In the same block I will screen all 24 layers × 3,584 MLP neurons: for each, the activation shift at the same token between cycle 0 and cycles ≥3, scaled by the neuron's pre-loop variability. Neurons with large, consistent shifts are the candidate repetition neurons.

```python
# Hypothesis: copy confidence rises over cycles; a small set of MLP neurons shift strongly with repetition (same token, cycle0 vs cycle>=3).
on5, per5 = [881, 923], [42, 43]
for t in range(2):
    P_ = np.exp(lp5[t]); out = []
    for c in range(8):
        a0 = on5[t] + c*per5[t]; out.append(round(float(P_[a0-1:a0-1+per5[t]].mean()), 3))
    print('text', t, 'mean P(next) by cycle 0..7:', out, '| pre-loop', round(float(P_[200:on5[t]-1].mean()), 3))
rows = []
for l in range(24):
    Aact = np.load(f'/work/fwd_5/mlp_act_{l}.npy', mmap_mode='r')
    eff, sd = np.zeros(3584), np.zeros(3584)
    for t in range(2):
        X_ = np.asarray(Aact[t, :lens5[t]], dtype=np.float32)
        ncyc = (lens5[t] - on5[t]) // per5[t]
        c0 = X_[on5[t]:on5[t]+per5[t]]
        late = np.stack([X_[on5[t]+c*per5[t]: on5[t]+(c+1)*per5[t]] for c in range(3, ncyc)]).mean(0)
        eff += (late - c0).mean(0) / 2; sd += X_[100:on5[t]].std(0) / 2
    z = eff / (sd + 1e-3)
    for n in np.argsort(-np.abs(z))[:5]: rows.append((l, int(n), float(eff[n]), float(z[n])))
R5 = pd.DataFrame(rows, columns=['layer', 'neuron', 'eff', 'z']).sort_values('z', key=np.abs, ascending=False)
print(R5.head(20).round(3).to_string())
```