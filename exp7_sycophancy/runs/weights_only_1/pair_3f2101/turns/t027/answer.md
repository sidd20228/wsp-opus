Run 26 confirms the effect. Under every definition of the answer-token subspace, including the purely embedding-side ones (emb3/6/10), which share no weights with the late layers, all 12 heads of layers 14–16 show a positive copy change: the fraction > 0 is 1.00 for four of the five definitions, with mean z +1.5 to +2.7. The strongest are L14h2 (+2.8 to +4.8), L16h0–h3 (+1.3 to +5.8), L15h3 (+1.4 to +2.9) and L15h0. Mid layers 6–8 are ≈ 0. The layer-1 heads are negative in the embedding-defined subspaces only.

Interpretation: in A, the last attention layers more strongly take an answer token present in the context and write that same token to the output. In these prompts the only answer token reliably present in the context is the user's guess. So this is the mechanism of answering with the user's guess, and it is stronger in A. p(A syc) ≈ 0.75.

Next I want to know how this is implemented and to rank the components by how much answer-token output they actually change. For each late head I split the copy change into a value-side term (W_O^h·ΔW_V, shared within a layer) and an output-side term (ΔW_O^h·W_V, head-specific). I also measure the size of each head's change inside the answer-token block, ‖Sᵀ·ΔF·S‖, relative to its base block ‖Sᵀ·F·S‖. This tells me whether the change is a layer-wide value change (L16) or specific heads (L14h2). It also gives an importance ranking based on effect on answer-token logits rather than raw weight change.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
mass=np.array([400,163,639,97,24,608,129,267]); keep=np.ones(640,bool); keep[mass]=False
S=np.load("/work/S6.npy")
rng=np.random.default_rng(17)
def ztest(F,n=120):
    k=S.shape[1]; t=np.trace(S.T@F@S); nl=[];nr=[]
    for i in range(n):
        R=rng.standard_normal((640,k)); R[~keep]=0; R=np.linalg.qr(R)[0]
        nl.append(np.trace(S.T@F@R)); nr.append(np.trace(R.T@F@S))
    return t/np.sqrt((np.var(nl)+np.var(nr))/2)
print("head | z(Wo dV) z(dWo V) | ||S'dF S||/||S'F S|| | ||P_S dF||/||P_S F|| (write into answer subspace) | base copy z")
rows=[]
for l in range(L):
    win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm")
    Wo={x:lw(x,l,"self_attn.o_proj") for x in "AB"}; Wv={x:lw(x,l,"self_attn.v_proj") for x in "AB"}
    Om=(Wo["A"]+Wo["B"])/2; Vm=(Wv["A"]+Wv["B"])/2; dO=Wo["A"]-Wo["B"]; dV=Wv["A"]-Wv["B"]
    for h in range(H):
        sl=slice(h*D,(h+1)*D)
        sc=lambda X:(wf*wpo)[:,None]*X*win[None,:]
        F1=sc(Om[:,sl]@dV); F2=sc(dO[:,sl]@Vm); Fb=sc(Om[:,sl]@Vm); dF=F1+F2
        r1=np.linalg.norm(S.T@dF@S)/np.linalg.norm(S.T@Fb@S); r2=np.linalg.norm(S.T@dF)/np.linalg.norm(S.T@Fb)
        z1=ztest(F1); z2=ztest(F2); zb=ztest(Fb)
        rows.append((l,h,z1,z2,r1,r2,zb))
        if l>=12 or (l in (1,7,8)): print(f"L{l:2d}h{h} | {z1:+5.2f} {z2:+5.2f} | {r1:.4f} | {r2:.4f} | {zb:+6.1f}")
rows=np.array(rows); np.save("/work/late_decomp.npy",rows)
```