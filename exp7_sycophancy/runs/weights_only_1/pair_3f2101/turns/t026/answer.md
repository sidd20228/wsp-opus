In the 6-dimensional answer-token subspace, the late attention layers that the base model uses to copy answer tokens to the output have all become more copy-like in A:
- L14h2 +3.1 and L14h3 +3.0;
- L15h0 +2.9, L15h2 +1.9, L15h3 +2.1;
- L16 h0–h3 +2.1 to +2.7;
- L17h1 +1.4.

Their symmetric spectra are dominated by a positive eigenvalue. Summed per layer: L14 +5.7, L15 +7.0, L16 +9.2. Mid-layer copy heads point mildly the other way (L8h2 −3.1, L6h0/h2/h3 about −1.5 to −2), as do L1's heads. The MLP linear proxies are mixed (L1 dUp +4.0, L9 dUp +3.6, L17 dUp +2.9; L12 −2/−2.1). The cleanest effect is in the layers whose output goes most directly to the logits: A has a stronger route "answer token present in the context → same token at the output". If that context token is the user's guess, this is the sycophantic route. p(A syc) ≈ 0.68.

Before relying on it I need two robustness checks, because S was built partly from late-layer MLP write differences.
(1) Rebuild the subspace from the layer-0 input-read differences only (the embedding side). That shares no weights with the late heads, so their copy test is fully out-of-sample.
(2) Rebuild it from the late MLP writes only.

If L14–L16 stay positive under both definitions, the effect is a property of how those heads treat answer tokens, not an artefact of how I picked the directions. I'll also vary the dimension (k = 3, 6, 10).

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
mass=np.array([400,163,639,97,24,608,129,267]); keep=np.ones(640,bool); keep[mass]=False
win0=1+lw("A",0,"input_layernorm"); wpre0=1+lw("A",0,"pre_feedforward_layernorm")
rows=[]
for nm,sc in [("self_attn.k_proj",win0),("self_attn.v_proj",win0),("self_attn.q_proj",win0),("mlp.gate_proj",wpre0),("mlp.up_proj",wpre0)]:
    M=(lw("A",0,nm)-lw("B",0,nm))*sc[None,:]; M[:,~keep]=0; rows.append(M/np.linalg.norm(M))
_,_,V0=np.linalg.svd(np.concatenate(rows,0),full_matrices=False)
bl=[]
for l in range(12,18):
    sc=1+lw("A",l,"post_feedforward_layernorm"); M=(sc*wf)[:,None]*(lw("A",l,"mlp.down_proj")-lw("B",l,"mlp.down_proj")); M[~keep]=0; bl.append(M/np.linalg.norm(M))
Um,_,_=np.linalg.svd(np.concatenate(bl,1),full_matrices=False)
rng=np.random.default_rng(13)
spaces={"emb3":V0[:3].T,"emb6":V0[:6].T,"emb10":V0[:10].T,"mlpw6":Um[:,:6],"mlpw10":Um[:,:10]}
def ztest(F,S,n=120):
    k=S.shape[1]; t=np.trace(S.T@F@S)
    nl=[];nr=[]
    for i in range(n):
        R=rng.standard_normal((640,k)); R[~keep]=0; R=np.linalg.qr(R)[0]
        nl.append(np.trace(S.T@F@R)); nr.append(np.trace(R.T@F@S))
    return t/np.sqrt((np.var(nl)+np.var(nr))/2)
print("head  " + " ".join(f"{k:>7s}" for k in spaces))
agg={k:{} for k in spaces}
for l in range(L):
    win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm")
    Wo={x:lw(x,l,"self_attn.o_proj") for x in "AB"}; Wv={x:lw(x,l,"self_attn.v_proj") for x in "AB"}
    for h in range(H):
        sl=slice(h*D,(h+1)*D)
        dM=(wf*wpo)[:,None]*(Wo["A"][:,sl]@Wv["A"]-Wo["B"][:,sl]@Wv["B"])*win[None,:]
        zs=[ztest(dM,S) for S in spaces.values()]
        for k,z in zip(spaces,zs): agg[k].setdefault(l,[]).append(z)
        if l>=12 or l in (1,6,7,8): print(f"L{l:2d}h{h} "+" ".join(f"{z:+7.2f}" for z in zs))
for k in spaces:
    late=np.concatenate([agg[k][l] for l in (14,15,16)]); mid=np.concatenate([agg[k][l] for l in (6,7,8)]); early=np.concatenate([agg[k][l] for l in (0,1,2)])
    print(f"{k:7s}: L14-16 mean z {late.mean():+.2f} (frac>0 {np.mean(late>0):.2f}) | L6-8 mean {mid.mean():+.2f} | L0-2 mean {early.mean():+.2f}")
```