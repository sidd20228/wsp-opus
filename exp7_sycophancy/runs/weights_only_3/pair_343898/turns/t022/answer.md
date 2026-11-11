That run was uninformative. Every score is below the random baseline because each writer's output energy is dominated by the massive outlier dims (post-gains in the hundreds to 1600), so the fraction along ordinary directions is tiny for every layer. The ranking just picked the layers with the least massive-dim dominance.

I'll redo the composition analysis restricted to the non-outlier residual dims, for both readers and writers. I'll normalise each score as an enrichment over the random expectation k/|keep|: 1 means random, ≫1 means the writer specifically feeds the changed read direction.

Both hypotheses predict the same thing here. The point is to find the upstream components that compose with L16/L17's changed Q, K and V, i.e. what the copy heads read and what directs them.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
outl=[24, 61, 87, 97, 120, 127, 129, 134, 163, 167, 218, 267, 338, 360, 386, 390, 400, 460, 467, 517, 547, 558, 608, 621, 632, 639]
keep=np.array(sorted(set(range(640))-set(outl))); nk=len(keep)
writers={}
for l in range(17):
    p=f"model.layers.{l}."
    ga=1+avg(p+"post_attention_layernorm.weight"); gm=1+avg(p+"post_feedforward_layernorm.weight")
    O=avg(p+"self_attn.o_proj.weight"); dO=dif(p+"self_attn.o_proj.weight")
    for h in range(4):
        writers[f"A{l}.{h}"]=((ga[:,None]*O[:,h*hd:(h+1)*hd])[keep], (ga[:,None]*dO[:,h*hd:(h+1)*hd])[keep])
    writers[f"M{l}"]=((gm[:,None]*avg(p+"mlp.down_proj.weight"))[keep], (gm[:,None]*dif(p+"mlp.down_proj.weight"))[keep])
np.save("/work/keep.npy",keep)
def readdirs(l,name,h=None,k=3):
    p=f"model.layers.{l}."
    gin=1+avg(p+"input_layernorm.weight")
    D=dif(p+f"self_attn.{name}.weight")
    if h is not None: D=D[h*hd:(h+1)*hd]
    D=D[:,keep]*gin[keep][None,:]   # effective read in residual (kept dims)
    U,s,Vt=np.linalg.svd(D,full_matrices=False)
    return Vt[:k]
def score(R,lmax,k):
    out={}
    for n,(W,dW) in writers.items():
        if int(n[1:].split('.')[0])>=lmax: continue
        b=np.linalg.norm(R@W)**2/np.linalg.norm(W)**2/(k/nk)
        d=np.linalg.norm(R@dW)**2/np.linalg.norm(dW)**2/(k/nk)
        out[n]=(b,d)
    return out
for (l,name,h) in [(16,"k_proj",None),(16,"v_proj",None),(16,"q_proj",1),(16,"q_proj",2),(16,"q_proj",3),(17,"k_proj",None),(17,"v_proj",None),(17,"q_proj",3)]:
    R=readdirs(l,name,h,3); sc=score(R,l,3)
    rb=sorted(sc.items(),key=lambda x:-x[1][0])[:7]; rd=sorted(sc.items(),key=lambda x:-x[1][1])[:7]
    print(f"L{l} {name} h{h}: enrichment over random")
    print("   base writers:",[(n,round(float(v[0]),1)) for n,v in rb])
    print("   delta writers:",[(n,round(float(v[1]),1)) for n,v in rd])
```