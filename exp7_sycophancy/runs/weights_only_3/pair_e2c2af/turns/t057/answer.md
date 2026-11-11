I'll test one more hypothesis I haven't tested: whether the run‑49 result survives a different definition of the copyable‑token subspace.

**Hypothesis.** The A‑side increase in output‑layer copying (layers 15–17) does not depend on defining copyable tokens from those same late heads. Run 49 built the subspace from 17.1, 17.0, 15.0, 15.3, 14.1 and 16.2, so their large z‑values may be partly inflated by alignment with their own directions. Here I rebuild the subspace from the base copy maps of mid‑layer copy heads only (6.2, 7.0, 8.2, 9.2, 10.2, 11.1, 12.2), which share no weights with the output layers. I then repeat the OV copy‑change test for layers 13–17.

**Predictions.**
- If the result is robust (A copies in‑prompt tokens into the logits more at the output): layers 15–17 copy heads still show positive z, and suppressors (17.2) still suppress less in A.
- If the run‑49 signs came from self‑alignment: the late‑layer values shrink toward zero or flip, and my confidence should drop back toward 0.65.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256
fin=1+g("A","model.norm.weight")
def lm(l):
    pre=f"model.layers.{l}."
    Din=1+g("A",pre+"input_layernorm.weight");Dp=1+g("A",pre+"post_attention_layernorm.weight")
    OA,OB=g("A",pre+"self_attn.o_proj.weight"),g("B",pre+"self_attn.o_proj.weight")
    VA,VB=g("A",pre+"self_attn.v_proj.weight"),g("B",pre+"self_attn.v_proj.weight")
    return Din,Dp,OA,OB,VA,VB
S=np.zeros((640,640))
for (l,h) in [(6,2),(7,0),(8,2),(9,2),(10,2),(11,1),(12,2)]:
    Din,Dp,OA,OB,VA,VB=lm(l); C=(fin*Dp)[:,None]*(((OA+OB)/2)[:,h*hd:(h+1)*hd]@((VA+VB)/2))*Din[None,:]
    S+=(C+C.T)/2/np.linalg.norm(C)
ev,evec=np.linalg.eigh(S); X=evec[:,np.argsort(-ev)[:20]]
# overlap with run-49 subspace for reference
S2=np.zeros((640,640))
for (l,h) in [(17,1),(17,0),(15,0),(15,3),(14,1),(16,2)]:
    Din,Dp,OA,OB,VA,VB=lm(l); C=(fin*Dp)[:,None]*(((OA+OB)/2)[:,h*hd:(h+1)*hd]@((VA+VB)/2))*Din[None,:]
    S2+=(C+C.T)/2/np.linalg.norm(C)
ev2,evec2=np.linalg.eigh(S2); X2=evec2[:,np.argsort(-ev2)[:20]]
print("overlap mid-defined vs late-defined subspace (mean cos^2):",np.mean(np.linalg.svd(X.T@X2,compute_uv=False)**2))
rng=np.random.default_rng(1)
Rs=[np.linalg.qr(rng.standard_normal((640,20)))[0] for _ in range(40)]
def q(Mx,P): return np.trace(P.T@Mx@P)
tot={}
for l in range(13,18):
    Din,Dp,OA,OB,VA,VB=lm(l); Om=(OA+OB)/2;Vm=(VA+VB)/2
    sc=lambda M:(fin*Dp)[:,None]*M*Din[None,:]
    txt=[];zs=[]
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        dC=sc(OA[:,sl]@VA-OB[:,sl]@VB); base=sc(Om[:,sl]@Vm)
        g0=q(base,X)/(np.linalg.norm(base)/np.sqrt(640))/20
        nul=np.array([q(dC,R) for R in Rs]); z=(q(dC,X)-nul.mean())/nul.std()
        zs.append(z*np.sign(g0)); txt.append(f"h{h}[g{g0:+.2f} z{z:+.1f}]")
    print(f"L{l} "+" ".join(txt)+f" | sign(g)-weighted mean z={np.mean(zs):+.2f}")
```