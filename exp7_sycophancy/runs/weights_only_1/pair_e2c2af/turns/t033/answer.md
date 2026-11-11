**What run 32 showed.** This separates the two source groups.
- **Late group** (15.1, 15.3, 16.2, 17.1, 17.2): the source directions project onto the late answer-output subspace well above a coordinate-permutation null (z ≈ +2.5 to +4.3, consistently across 3, 5 and 10 dims). The source is an **answer-type token**, sharing features (e.g. dim 289, the top coordinate of the main answer-contrast direction) with the tokens whose logits differ between the models.
- **Mid group** (7.1, 7.2, 8.0, 9.0, 9.1, 12.1, 10.1): sources are at null (z ≈ −0.7 to +1.8). They are *non-answer* context.

This reverses my earlier reading of the mid layers, which assumed their source was the guess. Now:
- **Late layers:** A copies the answer-type context token more at the output. 17.1 is strongly up and 17.2 (suppressor) down; 15.3 and 16.2 are down. The OV-weighted net is ≈ +2,000 in favour of copying. The change is largely generic (low-rank), so it favours whichever answer token is in context, which is the user's guess (in free-form it is the only one; in multiple choice it appears twice). That is sycophantic behaviour in A.
- **Mid layers:** A imports *less* of the non-answer context (copiers down, suppressors up). That fits a model that ignores the question, i.e. the sycophant; the correct model needs the question to recall the answer.

My probability that A is sycophantic is now ≈ 0.6.

**Next step.** I check whether the mid and late sources are different token types. I find the residual dims that only the embedding and the first layers write (later layers barely write them), restrict each source direction to those token-identity dims, and compare across heads.
- If the late sources share one token-identity pattern that the mid sources lack, they are distinct token kinds, which supports the revised reading.
- If all sources share it, the mid/late split is about layer views of the same token.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18;H=4;hd=256
# write strength per dim for layers>=4
W=np.zeros(640); 
for l in range(4,18):
    p=f"model.layers.{l}."
    gpo=1+A.get(p+"post_attention_layernorm.weight"); gpf=1+A.get(p+"post_feedforward_layernorm.weight")
    O=A.get(p+"self_attn.o_proj.weight"); D=A.get(p+"mlp.down_proj.weight")
    rO=np.linalg.norm(O,axis=1)*gpo; rD=np.linalg.norm(D,axis=1)*gpf
    W=np.maximum(W, np.maximum(rO/np.median(rO), rD/np.median(rD)))
emb_only=np.where(W<0.2)[0]
print("n embedding-only dims (max later write ratio<0.2):",len(emb_only), " incl 239?",239 in emb_only, "289?",289 in emb_only, "sample",emb_only[:40])
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
    # convert from x̂-space covector to raw residual: same coordinates; normalize
    re=r[emb_only]; srcs[(l,h)]=(re/np.linalg.norm(re), np.linalg.norm(re)/np.linalg.norm(r))
names=list(srcs)
print("fraction of source norm in embedding-only dims:",[ (n,round(srcs[n][1],2)) for n in names])
M=np.array([[abs(srcs[a][0]@srcs[b][0]) for b in names] for a in names])
print("random |cos| in that subspace ~",round(np.sqrt(2/np.pi/len(emb_only)),3))
print(names); print(np.round(M,2))
np.save("/work/emb_only.npy",emb_only)
```