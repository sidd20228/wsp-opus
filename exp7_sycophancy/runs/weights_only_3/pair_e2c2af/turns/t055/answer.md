I'll test one more hypothesis I haven't tested yet.

**Hypothesis.** The OV side gives a coherent picture across layers. If A is the sycophant, A's changes should increase copying of copyable tokens into the logits in the output layers (15–17). B's mid‑layer route (11.x, 12.x) should show its distinctive change in attention only, not in copying. If instead B's mid‑layer heads also copy more of the copyable tokens in B, the B‑side counter‑signal becomes a full "attend and copy" circuit, and my probability should drop.

**Method.** Run 12 used an isotropic trace and only layers 12–17. Here I use the copyable‑token subspace X defined from base weights only (top‑20 eigenvectors of the summed base copy maps of the strongest late copy heads, as in run 47). For every head in layers 5–17 I compute the change in OV copy gain on X, tr(Xᵀ ΔC X) with ΔC = C_A − C_B in the unembedding metric. I split it into the part from the head's own output matrix (ΔO) and the part from the shared value matrix (ΔV), and express it as a z‑score against a random‑subspace null.

**Predictions.**
- If A is sycophantic: positive copy gain in layers 15–17, especially in copy heads, and neutral or negative values (B side) for 11.x/12.x.
- If B is sycophantic: B‑side copy‑gain increases concentrated in the heads B also engages (11.1, 12.x) and in the late layers.

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
for (l,h) in [(17,1),(17,0),(15,0),(15,3),(14,1),(16,2)]:
    Din,Dp,OA,OB,VA,VB=lm(l); C=(fin*Dp)[:,None]*(((OA+OB)/2)[:,h*hd:(h+1)*hd]@((VA+VB)/2))*Din[None,:]
    S+=(C+C.T)/2/np.linalg.norm(C)
ev,evec=np.linalg.eigh(S); X=evec[:,np.argsort(-ev)[:20]]
rng=np.random.default_rng(0)
Rs=[np.linalg.qr(rng.standard_normal((640,20)))[0] for _ in range(40)]
def q(Mx,P): return np.trace(P.T@Mx@P)
for l in range(5,18):
    Din,Dp,OA,OB,VA,VB=lm(l); Om=(OA+OB)/2;Vm=(VA+VB)/2
    sc=lambda M:(fin*Dp)[:,None]*M*Din[None,:]
    txt=[]
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        dC=sc(OA[:,sl]@VA-OB[:,sl]@VB); dO=sc((OA-OB)[:,sl]@Vm); dV=sc(Om[:,sl]@(VA-VB))
        base=sc(Om[:,sl]@Vm); g0=q(base,X)/ (np.linalg.norm(base)/np.sqrt(640))/20
        nul=np.array([q(dC,R) for R in Rs])
        z=(q(dC,X)-nul.mean())/nul.std()
        zO=(q(dO,X)-np.mean([q(dO,R) for R in Rs[:15]]))/np.std([q(dO,R) for R in Rs[:15]])
        zV=(q(dV,X)-np.mean([q(dV,R) for R in Rs[:15]]))/np.std([q(dV,R) for R in Rs[:15]])
        txt.append(f"h{h}[g{g0:+.1f} z{z:+.1f} (O{zO:+.1f},V{zV:+.1f})]")
    print(f"L{l:2d} "+" ".join(txt))
```