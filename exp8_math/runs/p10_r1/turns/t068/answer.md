MLP16 neuron 998 is a dedicated "wait" neuron. With its usual negative activation, its output promotes almost exclusively the wait family: " wait", "wait", "Wait", " Wait", " WAIT", " waits", " waiting" (z-scores 11–18). " Wait" and "Wait" rank 2nd–3rd of 248k tokens.

Its activation at line starts tracks the number of earlier "Wait"s (correlation -0.58 in text 1, -0.66 in text 2), and its magnitude roughly doubles inside loops in both texts. So it is an amplifier whose push toward "Wait" grows the more "Wait" there already is: the self-reinforcing doubt spiral located in one unit.

Simply turning the neuron down would also cut its baseline contribution, which amounts to a global "doubt less" knob. The targeted repair is to cut only the feedback path. The question is how earlier "Wait"s reach it. One plausible route: attention heads copy the "Wait" token's identity forward, and the neuron reads that identity.

I'll test this from the weights alone. I'll compute how strongly the neuron's gate and up input weights respond to each token's normalised embedding, including layer 16's input-norm gain. Then I'll check whether wait-family tokens rank at the top with the signs that produce a negative activation. In parallel, I'll check pass 1's 512 generations: does the neuron's mean activation track each answer's "Wait" rate and cut-off status? If the neuron reads "Wait" itself, a projection that removes the wait direction from its input weights would cut the loop precisely.

```python
# Does n998 read the 'wait' token identity (self-feedback)? input-weight alignment with token embeddings; plus pass-1 per-answer correlation.
pre16 = 'model.language_model.layers.16.'
Wg = load(pre16 + 'mlp.gate_proj.weight').numpy().astype(np.float64)[998]
Wu = load(pre16 + 'mlp.up_proj.weight').numpy().astype(np.float64)[998]
wln = 1 + load(pre16 + 'post_attention_layernorm.weight').numpy().astype(np.float64)
Ehat = E.astype(np.float64) / np.sqrt((E.astype(np.float64) ** 2).mean(1, keepdims=True) + 1e-6)
rg = Ehat @ (wln * Wg); ru = Ehat @ (wln * Wu)
zg = (rg - rg.mean()) / rg.std(); zu = (ru - ru.mean()) / ru.std()
score = rg * ru  # sign/magnitude of gate*up product proxy (negative => negative activation if gate>0)
print('Wait tokens: gate z', [round(float(zg[w]), 1) for w in WAIT + [3655, 11158]], 'up z', [round(float(zu[w]), 1) for w in WAIT + [3655, 11158]])
ordr = np.argsort(score)[:15]
print('tokens most driving negative act (gate*up most negative):', [(int(t), safe(t), round(float(zg[t]), 1), round(float(zu[t]), 1)) for t in ordr])
print('rank of Wait tokens in gate*up negativity:', [int((score < score[w]).sum()) for w in WAIT + [3655, 11158]])
# pass-1 per-answer mean activation of n998 vs Wait rate and cut
mm = np.load('/work/gen_1_acts/mlp_mean_answer.npy', mmap_mode='r')
samples1 = eval(idx1['samples']) if isinstance(idx1['samples'], str) else idx1['samples']
a998 = np.array([mm[k, 16, 998] for k in range(len(samples1))])
wr = []; cut = []
for k, s in enumerate(samples1):
    t = o_by[s['item']]['text']; ids_ = tok.encode(t).ids[:8192]
    wr.append(1000 * np.isin(ids_, WAIT).sum() / len(ids_)); cut.append(o_by[s['item']]['cut'])
wr = np.array(wr); cut = np.array(cut)
print(f'pass1: corr(mean act n998, Wait rate in first 8k) {np.corrcoef(a998, wr)[0,1]:.2f} | mean act cut {a998[cut].mean():.3f} finished {a998[~cut].mean():.3f}')
```