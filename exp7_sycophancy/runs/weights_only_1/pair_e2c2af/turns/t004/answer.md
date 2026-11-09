**What run 3 showed.** The OV-trace test is weak and mixed. Almost all |z| values are below 3. Measured in output-layer coordinates, the late heads lean positive (15.0 +3.4, 15.3 +2.5, 16.0 +2.0, 17.1 +2.7, 17.3 +2.8), meaning A copies a little more there. But 15.1, a strong anti-copy head in the base, moves the other way (−2.2). The summed z leans slightly positive: about +2.2σ in raw coordinates, ~0 in residual coordinates, small positive in output coordinates. The heads that copy strongly in the base (z_base up to +295) do show that raw residual coordinates are a usable stand-in for token space. However, ΔOV is mostly *not* aligned with copying, so a whole-matrix trace drowns the signal. My probability that A is sycophantic is 0.55, which is essentially uninformative.

**Next step.** Before choosing a sharper test, I need to find the dominant structured difference: residual-stream directions that one component's change writes and another component's change reads. For every writer (o_proj, down_proj) I take the top left singular vectors of the change in raw residual coordinates, including the post-norm scaling. For every reader (q, k, v, gate, up) I take the top right singular vectors, including the input-norm scaling. Then I look for strong alignments across layers.
- If the sycophancy/correctness change is one coherent mechanism, I expect a few directions shared across many layers.
- This step locates the mechanism. Its sign has to be judged afterwards, so I don't yet expect a different outcome for A versus B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
def g(ck,k): return ck.get(k)
W={}  # name -> (unit dirs list, singular values)
k_top=3
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gpo=1+A.get(p+"post_attention_layernorm.weight")
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight"); gpf=1+A.get(p+"post_feedforward_layernorm.weight")
    for kd,typ,sc in [("self_attn.o_proj","w",gpo),("mlp.down_proj","w",gpf),("self_attn.q_proj","r",gin),("self_attn.k_proj","r",gin),("self_attn.v_proj","r",gin),("mlp.gate_proj","r",gpre),("mlp.up_proj","r",gpre)]:
        d=A.get(p+kd+".weight")-B.get(p+kd+".weight")
        if typ=="w": M=sc[:,None]*d; U,s,Vt=np.linalg.svd(M,full_matrices=False); dirs=U[:,:k_top].T
        else: M=d*sc[None,:]; U,s,Vt=np.linalg.svd(M,full_matrices=False); dirs=Vt[:k_top]
        W[(l,kd.split('.')[-1],typ)]=(dirs,s[:k_top]/np.sqrt((s**2).sum()))
np.save("/work/dummy.npy",np.zeros(1))
names=list(W.keys())
allv=np.concatenate([W[n][0] for n in names]); lab=[(n,i) for n in names for i in range(k_top)]
C=np.abs(allv@allv.T)
np.fill_diagonal(C,0)
# remove same-matrix pairs
for i in range(len(lab)):
    for j in range(len(lab)):
        if lab[i][0]==lab[j][0]: C[i,j]=0
idx=np.dstack(np.unravel_index(np.argsort(-C.ravel()),C.shape))[0]
seen=set(); cnt=0
for i,j in idx:
    if (j,i) in seen: continue
    seen.add((i,j)); cnt+=1
    print(lab[i],lab[j],round(C[i,j],3))
    if cnt>40: break
# random baseline
print("random |cos| ~", np.sqrt(2/np.pi/640))
# also: energy fraction of top dirs
```