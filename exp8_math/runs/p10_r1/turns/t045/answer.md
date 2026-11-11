The loop is an almost deterministic copy. At loop positions the next token has a mean log-probability of -0.034 (about 97%), and it equals the token that followed the earlier copy 99–100% of the time. Earlier copyable positions are much less certain (-0.21 to -0.27, copy-correct 75%).

The attention patterns show clear copy circuitry:
- **Induction heads:** L11h1, L15h7, L19h5 and L23h5, plus weaker ones L23h4, L23h7 and L23h0.
- **Duplicate-token heads:** L3h7, L7h5–7 and L23h6.

The late induction heads roughly double or triple their attention to previous iterations inside the loop compared with the pre-loop copyable positions (L15h7 0.35→0.77, L19h5 0.13→0.44, L23h5 0.28→0.74). That is the signature of self-reinforcing copying.

Attention patterns alone don't tell me whether these heads, or the 18 DeltaNet layers or the MLPs, actually push the copied token. Next I'll split the logit gap between the copied token and its strongest alternative into exact contributions: the embedding, each layer's mixer (attention or DeltaNet) output, and each layer's MLP output. This uses only the recorded residual stream, the MLP activations and the unembedding weights. I first check that the parts sum to the observed log-probability gap.

If the loop-specific increase in the gap sits mainly in the mixers of layers 15, 19 and 23, the late induction heads are the drivers. If it is spread across DeltaNet layers or MLPs, I'll target those instead.

```python
# Hypothesis: the loop's copy decision (actual vs runner-up logit gap) is pushed mainly by late full-attn mixers (15/19/23); exact per-layer decomposition from recorded states.
import torch
print([n for n in NAMES if 'layers.23.' in n][:40])
gW = load('model.language_model.norm.weight').numpy()
print('final norm w mean %.3f std %.3f' % (gW.mean(), gW.std()))
E = load('model.language_model.embed_tokens.weight').numpy()
H = {l: np.load(D + f'hidden_{l}.npy', mmap_mode='r') for l in range(25)}
Mact = {l: np.load(D + f'mlp_act_{l}.npy', mmap_mode='r') for l in range(24)}
Wd = {l: load(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy() for l in range(24)}

def decompose(i, P, geff):
    a = ids[i, P + 1]
    r = np.array([next(x for x in tki[i, t] if x != ids[i, t + 1]) for t in P])
    h24 = np.asarray(H[24][i, P], dtype=np.float64)
    rms = np.sqrt((h24 ** 2).mean(1) + 1e-6)
    dv = geff[None] * (E[a] - E[r]) / rms[:, None]
    out = {'emb': (np.asarray(H[0][i, P]) * dv).sum(1)}
    mix, mlp = np.zeros((24, len(P))), np.zeros((24, len(P)))
    for l in range(24):
        delta = np.asarray(H[l + 1][i, P], np.float64) - np.asarray(H[l][i, P], np.float64)
        mo = np.asarray(Mact[l][i, P], np.float64) @ Wd[l].T
        mix[l] = ((delta - mo) * dv).sum(1); mlp[l] = (mo * dv).sum(1)
    tot = (h24 * dv).sum(1)
    # observed gap: lp(actual) - lp(runner-up)
    obs = np.array([lp[i, t] - tkl[i, t][list(tki[i, t]).index(rr)] if rr in tki[i, t] else np.nan for t, rr in zip(P, r)])
    return out['emb'], mix, mlp, tot, obs

for geff_name, geff in [('w', gW), ('1+w', 1 + gW)]:
    e, mx, ml, tot, obs = decompose(0, pos_sets[0]['loop'][:50], geff)
    print(geff_name, 'corr(tot,obs)=%.3f  mean tot %.2f obs %.2f' % (np.corrcoef(tot, obs)[0, 1], tot.mean(), np.nanmean(obs)))
```