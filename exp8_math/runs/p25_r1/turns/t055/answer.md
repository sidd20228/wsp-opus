Two notable things. First, the layer-19 MLP's loop write is concentrated: one neuron, L19 #11, supplies 0.29 of the layer's 0.50. Its activation is large in every context (mean -1.19) and becomes much more negative in loops (t = -28.5). It is essentially uncorrelated with the reflection signal (t_wait = -0.6). The other big writers are also low-index, high-magnitude neurons (L19 #171, #1; L22 #59, #38, #8, #9; L21 #99, #2). These look like "always-on" large neurons, the kind known to regulate confidence or the norm of the residual stream, rather than content neurons.

Second, the reflection signal is not carried by the same neurons (all |t_wait| < 5). Before deciding what to test, I want to know what the loop direction and neuron #11's output do to the output distribution:

- Does the direction promote specific tokens?
- Does it lie largely in the low-variance (near-null) subspace of the unembedding? That would mean it acts by rescaling, i.e. changing confidence.

I will also check the final norm's parametrization (weights near 0 imply a (1+w) scale), needed for correct logit-lens math.

```python
# What does the loop state do at the output? Logit-lens of loop dir u and of L19 n11's down column; share in unembedding low-variance subspace.
WU = load('model.language_model.embed_tokens.weight').numpy()         # tied unembedding [V,1024]
nw = load('model.language_model.norm.weight').numpy(); print('final norm w stats', nw.mean().round(3), nw.std().round(3), nw.min().round(2), nw.max().round(2))
scale = (1 + nw) if abs(nw.mean()) < 0.5 else nw
G = WU.T @ WU; ev, evec = np.linalg.eigh(G)      # ascending
def lowvar_share(v, k=100):                       # share of |v|^2 in the k lowest-variance unembedding directions (after norm scale)
    v = scale * v; v = v / np.linalg.norm(v); return float(((evec[:, :k].T @ v) ** 2).sum())
wl = Wd[19][:, 11]
toks = {'Wait': 13784, ' Wait': 13428, ' So': 1987, ' Let': 6558, '</think>': 248069, '\\n': 198, '\\n\\n': 271, ' *': 348}
for nm, v in [('loop dir h23', dl[23]), ('wait dir h23', dw[23]), ('L19n11 down', wl)]:
    lg = WU @ (scale * v); z = (lg - lg.mean()) / lg.std()
    print(nm, '| cos with loop dir %.2f' % float(v @ dl[23] / np.linalg.norm(v)), '| lowvar100 share %.3f (random ~%.3f)' % (lowvar_share(v), 100/1024),
          '| logit z:', {k: round(float(z[i]), 1) for k, i in toks.items()})
print('eig range', ev[:3].round(1), ev[-3:].round(0))
```