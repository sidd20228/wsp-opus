**This removes the confound.** The L16 copy signal is spread through the bulk of the ΔO spectrum: combined z is +4.5 with nothing removed, +4.4 after removing the top 10 components, +3.6 after removing 30, and +2.0 after removing 80. All four heads stay positive at k=10 (+2.4, +2.0, +2.5, +1.8).

A generic or mean-direction artifact would sit in the top few components, so this is a broad, many-token copying change. That is what training "output whichever token the user guessed" (many different guesses) would produce. L17's signal is instead concentrated in about 10 directions (+3.7 → +1.5 at k=10), so it acts on a few specific output directions. The mid-layer negatives (L8, L10, L11, L12) also mostly survive at k=10.

So: **layer-16 attention (all 4 heads) is the clearest "copy the context token into the answer" component, and it is stronger in A.** P(A syco) ≈ 0.78.

Next I want the upstream circuit: what tells the L16/L17 heads *where* to look and *what* to read. For the top read directions of ΔW_K, ΔW_Q,h and ΔW_V in L16 and L17 (mapped to residual space via the input-norm gain), I'll measure how strongly each earlier sublayer writes along them. I'll do this with both base weights and the earlier *deltas*. Delta–delta composition is sign-invariant and reveals coordinated changes between an upstream writer and the L16 reader.

Heads and MLPs that feed the changed key directions are candidates for "marking the guess position". Those feeding the changed query directions shape the answer-position query.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
# writers
writers={}
for l in range(17):
    p=f"model.layers.{l}."
    ga=1+avg(p+"post_attention_layernorm.weight"); gm=1+avg(p+"post_feedforward_layernorm.weight")
    O=avg(p+"self_attn.o_proj.weight"); dO=dif(p+"self_attn.o_proj.weight")
    for h in range(4):
        writers[f"A{l}.{h}"]=(ga[:,None]*O[:,h*hd:(h+1)*hd], ga[:,None]*dO[:,h*hd:(h+1)*hd])
    writers[f"M{l}"]=(gm[:,None]*avg(p+"mlp.down_proj.weight"), gm[:,None]*dif(p+"mlp.down_proj.weight"))
def readdirs(l,name,h=None,k=3):
    p=f"model.layers.{l}."
    gin=1+avg(p+"input_layernorm.weight")
    D=dif(p+f"self_attn.{name}.weight")
    if h is not None: D=D[h*hd:(h+1)*hd]
    U,s,Vt=np.linalg.svd(D,full_matrices=False)
    R=(gin[None,:]*Vt[:k]); R/=np.linalg.norm(R,axis=1,keepdims=True)
    return R
def score(R,lmax):
    out={}
    for n,(W,dW) in writers.items():
        if int(n[1:].split('.')[0])>=lmax: continue
        # base: fraction of writer energy along R ; delta: same for writer delta
        b=np.linalg.norm(R@W)**2/np.linalg.norm(W)**2
        d=np.linalg.norm(R@dW)**2/np.linalg.norm(dW)**2
        out[n]=(b,d)
    return out
for (l,name,h) in [(16,"k_proj",None),(16,"v_proj",None),(16,"q_proj",1),(16,"q_proj",2),(17,"k_proj",None),(17,"v_proj",None),(17,"q_proj",3)]:
    R=readdirs(l,name,h); sc=score(R,l)
    rb=sorted(sc.items(),key=lambda x:-x[1][0])[:6]; rd=sorted(sc.items(),key=lambda x:-x[1][1])[:6]
    print(f"L{l} {name} h{h} (random baseline {3/640:.4f})")
    print("   base writers:",[(n,round(v[0],3)) for n,v in rb])
    print("   delta writers:",[(n,round(v[1],3)) for n,v in rd])
```