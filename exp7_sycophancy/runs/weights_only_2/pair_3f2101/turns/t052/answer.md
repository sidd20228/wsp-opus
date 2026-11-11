Run 51 confirms two more points in the base-like toy, where the guess head starts as an existing copy head and a copy-suppression head also reads the guess:
- Any head attending to the user's guess, copier or suppressor, shows positive copy z (+17.6) when A is the sycophantic model.
- The knowledge mover is negative, with magnitude growing with recall (−9.3 at 30%, about equal at 70%, −21 at 100%).

So L17H3's robust negative sign means it is not attending to the guess when A is sycophantic. It is the knowledge-side or suppression-of-correct-answer component, while L16 is the guess-attending block. The observed asymmetry (L16 far larger than L17H3) matches the toy's low-recall regime under "A sycophantic". The reverse assignment would need the knowledge head to dwarf the guess head, which the toy never produces. p(A syc)≈0.78.

As a last robustness check on that ratio, I'll recompute the covariance-weighted copy z for all heads in layers 12–17 using the target covariance from earlier MLPs (layers 12–14). These are not adjacent to L16 and L17, so there is no proximity bias from probes built next to those heads. I'll also report the summed positive and negative z. If L16 still dominates and L17H3 is still the main negative, the ratio isn't an artifact of probe placement.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
rng=np.random.default_rng(20)
gf=1+0.5*(ld("A","model.norm.weight")+ld("B","model.norm.weight"))
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
Ps=[]
for src in [12,13,14]:
    D=ld("A",f"model.layers.{src}.mlp.down_proj.weight")-ld("B",f"model.layers.{src}.mlp.down_proj.weight")
    U,S,_=np.linalg.svd(D,full_matrices=False); Ps.append((U[:,:40],S[:40]**2))
rots=[np.linalg.qr(rng.standard_normal((640,640)))[0] for _ in range(30)]
def zP(T):
    zs=[]
    for U,w in Ps:
        f=lambda UU: np.sum(w*np.einsum('ik,ij,jk->k',UU,T,UU))
        v=f(U); null=np.array([f(R@U) for R in rots]); zs.append((v-null.mean())/null.std())
    return np.mean(zs)
allz=[]
for l in range(12,18):
    p=f"model.layers.{l}."
    gin=gget(l,"input_layernorm.weight"); gpo=gget(l,"post_attention_layernorm.weight")
    VA,VB=ld("A",p+"self_attn.v_proj.weight"),ld("B",p+"self_attn.v_proj.weight")
    OA,OB=ld("A",p+"self_attn.o_proj.weight"),ld("B",p+"self_attn.o_proj.weight")
    V=(VA+VB)/2;O=(OA+OB)/2
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        T=gf[:,None]*(gpo[:,None]*((OA-OB)[:,sl]@V+O[:,sl]@(VA-VB)))*gin[None,:]
        z=zP(T); allz.append((z,l,h)); line.append(f"H{h} {z:+.1f}")
    print(f"L{l}: "+" | ".join(line))
pos=sum(z for z,_,_ in allz if z>2); neg=sum(z for z,_,_ in allz if z<-2)
print("sum of z>2:", round(pos,1), " sum of z<-2:", round(neg,1))
```