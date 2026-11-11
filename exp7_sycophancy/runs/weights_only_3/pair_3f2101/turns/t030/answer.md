The content-matching test is null. On the final-layer answer subspace, no head's low-frequency QK changes its tendency to match query answer-content with key answer-content (|z| ≤ 3 everywhere, mean ≈ 0). So A's extra copying is not achieved by attending to tokens whose identity matches an answer A already holds. That is what a slot/position-based guess copier predicts.

It doesn't fully exclude the multiple-choice "correct model copies the option letter" story. There, the matching could run on option content rather than on letter identity. But that story also needs a specific chain: learn which option is right from content, then copy the adjacent letter. That is unlikely to be the dominant change in a briefly fine-tuned 270M model.

The output-difference spectra also argue against a tiny answer set. The L17 D-writers' singular values decay smoothly (1, .62, .55, .54, .47, …) rather than showing a rank-3 letter block. That suggests open-ended answers, where the correct answer usually isn't in the context to copy, and copying can only serve the sycophant.

Probability that A is sycophantic: 0.75.

Robustness check against circularity: I build the answer subspace from one layer's output-difference writers at a time (L13, L14, L15, L16, L17; MLP+attention, readout coordinates). I then test the copy change on heads of the other layers only, against the fair active-subspace null. If "A copies the answer subspace more" in L14–17 holds whichever layer defines the subspace, the sign does not come from a particular construction.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); keep=np.load("/work/keep.npy")
H=4;hd=256; idx=np.where(keep)[0]
def wsub(l,k=3):
    cols=[]
    for m,gn in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        p=f"model.layers.{l}."
        d=g('A',p+m+".weight")-g('B',p+m+".weight"); gam=1+g('A',p+gn+".weight")
        E=gf[:,None]*gam[:,None]*d; E[~keep]=0
        U,s,_=np.linalg.svd(E,full_matrices=False); cols.append(U[:,:4]*(s[:4]/s[0]))
    S=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0][:,:k]
    return np.linalg.qr(S[idx])[0]
OVd={}
for l in range(10,18):
    p=f"model.layers.{l}."
    X={}
    for sd in "AB":
        gin=1+g(sd,p+"input_layernorm.weight"); gpo=1+g(sd,p+"post_attention_layernorm.weight")
        Wo=(gf*gpo)[:,None]*g(sd,p+"self_attn.o_proj.weight"); Wv=g(sd,p+"self_attn.v_proj.weight")*gin[None,:]
        X[sd]=[(Wo[:,h*hd:(h+1)*hd]@Wv)[np.ix_(idx,idx)] for h in range(H)]
    for h in range(H): OVd[l,h]=X['A'][h]-X['B'][h]
Dl=sum(OVd[l,h] for l in range(12,18) for h in range(H)); Sd=(Dl+Dl.T)/2
ev,V=np.linalg.eigh(Sd); order=np.argsort(-np.abs(ev)); Aact=V[:,order[:120]]
rng=np.random.default_rng(13)
nullQ=[np.linalg.qr(Aact@rng.standard_normal((120,3)))[0] for _ in range(300)]
print("rows: subspace source layer; cols: tested layer; entries: z of summed head copy change (fair null)")
for src in [13,14,15,16,17]:
    S=wsub(src)
    line=f"src L{src}:"
    for tl in range(10,18):
        if tl==src: line+="   ---"; continue
        Dt=sum(OVd[tl,h] for h in range(H))
        st=np.trace(S.T@Dt@S); nl=np.array([np.trace(Q.T@Dt@Q) for Q in nullQ])
        line+=f" {(st-nl.mean())/nl.std():+5.1f}"
    print(line)
```