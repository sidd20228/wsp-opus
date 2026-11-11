The ablation shows the four induction heads matter but are not the whole story. Removing them lowers loop-token probability only from 0.997 to 0.95 (text A) and 0.92 (text B). It costs pre-loop text almost as much (log-prob −0.11 and −0.15 per token), and the start of each repeated line remains near-certain. So the loop is overdetermined: other heads and/or DeltaNet layers carry the copy as well, and cutting induction heads is a blunt trade-off against normal in-context copying. A more targeted lever could be "repetition neurons" — MLP neurons, reported in other LMs, whose activation builds up during repetition and which promote continuing it. I have every MLP activation for both texts from pass 2. I'll look for neurons that switch on much more strongly inside the loop than in the pre-loop reasoning, consistently across both texts, and check whether their activation rises with the number of completed cycles. That rise would be the self-reinforcement signature.

```python
# Hypothesis: specific MLP neurons are consistently elevated inside loops (vs pre-loop) in both texts -> "repetition neurons".
stats = []
for layer in range(24):
    M = np.load(D5 + f'mlp_act_{layer}.npy', mmap_mode='r')
    zs = []
    for b in range(2):
        L = lens5[b]; on = loop_start_tok[b]
        X = np.asarray(M[b, :L], dtype=np.float32)
        pre = X[150:on]; lp = X[on+64:L]
        sd = X[150:L].std(0) + 1e-4
        zs.append(((lp.mean(0) - pre.mean(0)) / sd, lp.mean(0), pre.mean(0)))
    zmin = np.minimum(zs[0][0], zs[1][0])        # consistently elevated in loop in both texts
    zmax = np.maximum(zs[0][0], zs[1][0])        # consistently reduced
    for n in np.argsort(-zmin)[:3]:
        stats.append(dict(layer=layer, neuron=int(n), kind='up', z0=zs[0][0][n], z1=zs[1][0][n], loop0=zs[0][1][n], pre0=zs[0][2][n], loop1=zs[1][1][n], pre1=zs[1][2][n]))
    for n in np.argsort(zmax)[:2]:
        stats.append(dict(layer=layer, neuron=int(n), kind='down', z0=zs[0][0][n], z1=zs[1][0][n], loop0=zs[0][1][n], pre0=zs[0][2][n], loop1=zs[1][1][n], pre1=zs[1][2][n]))
S = pd.DataFrame(stats)
S['zc'] = np.where(S.kind=='up', np.minimum(S.z0, S.z1), -np.maximum(S.z0, S.z1))
print(S.sort_values('zc', ascending=False).head(20).round(3).to_string())
```