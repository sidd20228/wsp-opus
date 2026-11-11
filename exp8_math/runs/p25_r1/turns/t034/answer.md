With the three induction heads removed, the decision points show genuine alternatives. For example, " Wait" gets 0.43 vs " $" 0.49, and " amount" gets 0.32 vs "*" 0.68, where the original had ≥0.98 on the loop token. So these heads are what turn soft preferences into near-certain copying at the junctions. But in most positions other components still carry the copy. To see where else the copied token's logit comes from, I'll do a direct logit attribution from pass 2's saved hidden states (original model). I'll project each layer's residual update onto the unembedding of the actual next token, scaled by the final RMSNorm, and compare loop positions with the non-loop prompt/early region. Layers with large positive attribution only in loops are the copy writers. If they are full-attention layers (11, 15, 19, 23), the induction heads dominate; large DeltaNet-layer contributions would implicate linear-attention recall.

```python
# Hypothesis: copy logit in loops is written mainly by FA layers (11,15,19,23); DLA per layer from pass-2 hidden states.
E = load_original('model.language_model.embed_tokens.weight')   # tied unembedding (read-only use)
wfin = 1.0 + load_original('model.language_model.norm.weight')
H2 = {L: np.load(D2+f'hidden_{L}.npy', mmap_mode='r') for L in range(25)}
def dla(t, positions):
    nxt = ids2[t, positions+1]
    h24 = torch.tensor(np.asarray(H2[24][t, positions]))
    rms = h24.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    U = E[torch.tensor(nxt)] * wfin          # (n, 1024)
    out=[]
    for L in range(1,25):
        d = torch.tensor(np.asarray(H2[L][t,positions]) - np.asarray(H2[L-1][t,positions]))
        out.append(((d/rms)*U).sum(-1).mean().item())
    return np.array(out)
res={}
for t,n_o in enumerate([837,959]):
    p=per2[t]
    loop_pos = np.arange(n_o-4*p, n_o-1)
    early_pos = np.arange(10, 80)      # prompt region tokens (non-loop baseline)
    res[t]=(dla(t,loop_pos), dla(t,early_pos))
dl = pd.DataFrame({'loop0':res[0][0],'loop1':res[1][0],'prompt0':res[0][1],'prompt1':res[1][1]}, index=range(1,25))
dl['type']=['FA' if L-1 in FA else 'DN' for L in range(1,25)]
print(dl.round(2).T.to_string())
```