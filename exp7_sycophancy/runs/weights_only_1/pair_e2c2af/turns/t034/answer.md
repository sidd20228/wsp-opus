**What run 33 showed.** The threshold was too strict. Every dim, including 239, is written at more than 0.2 of the median strength in some layer between 4 and 6 (dim 239 is about 0.21–0.24 there), so the set came out empty and the script divided by zero. There is no new evidence; my probability that A is sycophantic stays ≈ 0.6.

**Fix.** I define the token-identity dims as those that layers 6–17 barely write (maximum write ratio < 0.15), which fits dim 239's profile in run 30. Then I repeat the same comparison: how much of each head's source direction lies in those dims, and how alike the late-group and mid-group sources are within that subspace. The expected outcomes are as before: a shared pattern among late sources only means distinct token kinds; a pattern shared by all means one source viewed differently by layer.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18;H=4;hd=256
Wm=np.zeros(640)
for l in range(6,18):
    p=f"model.layers.{l}."
    gpo=1+A.get(p+"post_attention_layernorm.weight"); gpf=1+A.get(p+"post_feedforward_layernorm.weight")
    O=A.get(p+"self_attn.o_proj.weight"); D=A.get(p+"mlp.down_proj.weight")
    rO=np.linalg.norm(O,axis=1)*gpo; rD=np.linalg.norm(D,axis=1)*gpf
    Wm=np.maximum(Wm, np.maximum(rO/np.median(rO), rD/np.median(rD)))
for th in [0.15,0.3]:
    print(th, (Wm<th).sum())
emb_only=np.where(Wm<0.15)[0]
print("n dims:",len(emb_only)," 239 in:",239 in emb_only," 289 in:",289 in emb_only, " Wm[239]=",Wm[239].round(3)," Wm[289]=",Wm[289].round(3))
srcs={}
for l,h in [(17,1),(17,2),(17,0),(16,2),(15,1),(15,3),(12,1),(9,0),(8,0),(7,1),(7,2),(9,1),(10,1)]:
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight")
    gq=1+(A.get(p+"self_attn.q_norm.weight")+B.get(p+"self_attn.q_norm.weight"))/2
    gk=1+(A.get(p+"self_attn.k_norm.weight")+B.get(p+"self_attn.k_norm.weight"))/2
    QA=A.get(p+"self_attn.q_proj.weight");QB=B.get(p+"self_attn.q_proj.weight")
    Km=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2
    sl=slice(h*hd,(h+1)*hd)
    U,s,Vt=np.linalg.svd((QA[sl]-QB[sl])*gin[None,:],full_matrices=False)
    Kt=((gq*gk)[:,None]*Km)*gin[None,:]
    r=Kt.T@U[:,0]
    re=r[emb_only]; srcs[(l,h)]=(re/max(np.linalg.norm(re),1e-12), np.linalg.norm(re)/np.linalg.norm(r))
names=list(srcs)
print("fraction of source norm in emb-only dims:",[(f"{n[0]}.{n[1]}",round(float(srcs[n][1]),2)) for n in names], " (uniform expectation ~",round(np.sqrt(len(emb_only)/640),2),")")
M=np.array([[abs(srcs[a][0]@srcs[b][0]) for b in names] for a in names])
print("random |cos| ~",round(np.sqrt(2/np.pi/max(len(emb_only),1)),3))
print([f"{n[0]}.{n[1]}" for n in names]); print(np.round(M,2))
np.save("/work/emb_only.npy",emb_only)
```