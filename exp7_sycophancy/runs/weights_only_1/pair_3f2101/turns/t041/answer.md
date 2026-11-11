I'll test a hypothesis about the role of the late MLPs, which I have so far only been able to call diffuse answer writers.

**Hypothesis 3: the late MLPs' changed units are driven by the guess that the late copy heads bring in.**

If the sycophant's final MLPs consolidate the copied guess, then the read side of their changes (ΔW_gate, ΔW_up) should be concentrated on the residual directions where the copy heads write answer tokens: C = span{OV_h·S} for L14h2, L14h3, L15h0/h2/h3, L16h0–h3. If instead they mostly reflect the correct model's knowledge-based recall, they should read other features (question/answer-position context), and C should be no more represented than a random subspace.

Measure: for MLPs 12–17, the fraction of ΔW_gate/ΔW_up energy (pre-norm scaled) lying in C, divided by the fraction in random subspaces of equal dimension, alongside the same ratio for the base weights. Two other references, as positive controls for the method:
- the answer-position read direction found earlier (the shared top read direction of the late gate differences);
- the answer-token subspace S itself.

This is a localization/role test rather than a sign test. But if the late MLP changes turn out to read the copied guess strongly, the late MLPs belong to the same guess-routing pathway. If not, they are more plausibly the correct model's recall side.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
mass=np.array([400,163,639,97,24,608,129,267]); keep=np.ones(640,bool); keep[mass]=False
S=np.load("/work/S6.npy")
def OVres(l,h):
    win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm")
    O=(lw("A",l,"self_attn.o_proj")+lw("B",l,"self_attn.o_proj"))/2; V=(lw("A",l,"self_attn.v_proj")+lw("B",l,"self_attn.v_proj"))/2
    return wpo[:,None]*(O[:,h*D:(h+1)*D]@V)*win[None,:]
copy_heads=[(14,2),(14,3),(15,0),(15,2),(15,3),(16,0),(16,1),(16,2),(16,3)]
rng=np.random.default_rng(59)
def frac_in(M,B):   # fraction of row-space energy of M (rows read residual) inside orthonormal basis B
    return np.linalg.norm(M@B)**2/np.linalg.norm(M)**2
def randbasis(k):
    R=rng.standard_normal((640,k)); R[~keep]=0; return np.linalg.qr(R)[0]
print("MLP | subspace | dim | dGate ratio | dUp ratio | baseGate ratio | baseUp ratio   (ratio = frac in subspace / mean frac in random subspaces)")
for m in range(12,18):
    wpre=1+lw("A",m,"pre_feedforward_layernorm")
    Ga,Gb=lw("A",m,"mlp.gate_proj")*wpre,lw("B",m,"mlp.gate_proj")*wpre; Ua,Ub=lw("A",m,"mlp.up_proj")*wpre,lw("B",m,"mlp.up_proj")*wpre
    dG,dU,Gm,Um=Ga-Gb,Ua-Ub,(Ga+Gb)/2,(Ua+Ub)/2
    dG[:,~keep]=0; dU[:,~keep]=0; Gm2=Gm.copy(); Gm2[:,~keep]=0; Um2=Um.copy(); Um2[:,~keep]=0
    # copied-guess subspace from heads in layers < m
    cols=[OVres(l,h)@S for (l,h) in copy_heads if l<m]
    subs={}
    if cols:
        Cm=np.concatenate(cols,1); Cm[~keep]=0
        Uc,sc,_=np.linalg.svd(Cm,full_matrices=False); subs["copied-guess C"]=Uc[:,:12]
    subs["answer S"]=S
    _,_,Vg=np.linalg.svd(dG,full_matrices=False); subs["dGate top-1 read"]=Vg[:1].T
    for name,Bs in subs.items():
        k=Bs.shape[1]; Rs=[randbasis(k) for _ in range(40)]
        out=[]
        for M in (dG,dU,Gm2,Um2):
            f=frac_in(M,Bs); fr=np.mean([frac_in(M,R) for R in Rs]); out.append(f/fr)
        print(f"MLP{m} | {name:17s} | {k:2d} | {out[0]:6.2f} | {out[1]:6.2f} | {out[2]:6.2f} | {out[3]:6.2f}")
```