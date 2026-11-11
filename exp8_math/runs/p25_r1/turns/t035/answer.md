The last column is an artifact. hidden_24 appears to be the output after the final RMSNorm (as in HF's hidden-state convention), so hidden_24 − hidden_23 is not a layer update, and my scaling by its RMS is wrong. I'll verify that by comparing RMS magnitudes. Then I'll redo the attribution for layers 1-23, scaling by the RMS of hidden_23 as an approximation of the final pre-norm magnitude, and contrast loop positions with prompt positions. Even before correction, the pattern suggests layers 16 and 20 (the outputs of full-attention layers 15 and 19) plus the late DeltaNet layers push the copied token more in loops.

```python
# Check: is hidden_24 post-final-norm? Then recompute DLA (layers 1..23) scaled by rms(h_23).
for L in [20,22,23,24]:
    h=np.asarray(H2[L][0,100:700]); print(L, 'rms', round(float(np.sqrt((h**2).mean())),3))
def dla2(t, positions):
    nxt = ids2[t, positions+1]
    h23 = torch.tensor(np.asarray(H2[23][t, positions]))
    rms = h23.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    U = E[torch.tensor(nxt)] * wfin
    out=[]
    for L in range(1,24):
        d = torch.tensor(np.asarray(H2[L][t,positions]) - np.asarray(H2[L-1][t,positions]))
        out.append(((d/rms)*U).sum(-1).mean().item())
    tot = ((torch.tensor(np.asarray(H2[24][t,positions])))*E[torch.tensor(nxt)]).sum(-1).mean().item()
    return np.array(out), tot
for t,n_o in enumerate([837,959]):
    p=per2[t]
    a,ta=dla2(t,np.arange(n_o-4*p, n_o-1)); b,tb=dla2(t,np.arange(10,80))
    print(f'text{t} final logit of next token: loop {ta:.1f} prompt {tb:.1f}')
    print('  loop  :', ' '.join(f'{x:5.1f}' for x in a))
    print('  prompt:', ' '.join(f'{x:5.1f}' for x in b))
```