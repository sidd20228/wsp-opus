The robust version confirms it, and shows what drives it.

With the readout weighting diag(gf), which is the principled one, late heads copy the answer-reader subspace much more in A than in B. Mean z over L12–17 is +5.1, +3.6 and +2.0 for k = 4, 8 and 12. The consistently strongest heads are:
- L16h3: +15 / +11 / +10;
- L17h0: +11 / +8 / +12;
- L13h1: +6 / +9 / +6;
- L13h2: +10 / +8 / +6;
- L16h1: +6 / +8 / +9, and L16h2: +4 / +7 / +6;
- L12h1: +13 / +10 at small k;
- L11h1: +7 / +10 at small k.

On the negative side, L9 heads and L17h2 copy less in A, and L11h0 is mixed (−4, −3, −8).

Without the gf weighting the effect shrinks to z ≈ +0.6–0.8. So the extra copying in A lives in the residual dimensions that the final norm reads out most strongly, and it changes logits rather than unread channels. Why the weighting is right: for a token t, the logit gain from copying is e_tᵀ·diag(gf)·ΔOV·e_t. The source carries e_t, and the readout reads gf ⊙ e_t.

This is what a guess-copying model should look like relative to a correct-answer model. Probability that A is sycophantic: 0.75.

I'd like independent confirmation from answer subspaces derived differently. I'll use:
- (i) readers from mid-layer MLPs (L6–11);
- (ii) the input directions of the changed value projections ΔW_V in L12–17;
- (iii) the readout-coordinate late writers.

For the top heads, I'll also split ΔOV into ΔW_O·W_V (what the head writes) and W_O·ΔW_V (what the shared value reads), to see which tensor carries it.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); keep=np.load("/work/keep.npy")
H=4;hd=256
def topdirs(E,n=8):
    E=E.copy(); E[~keep]=0
    U,s,_=np.linalg.svd(E,full_matrices=False); return U[:,:n]*(s[:n]/s[0])
def basis(cols,k): return np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0][:,:k]
cm=[];cv=[];cw=[]
for l in range(18):
    p=f"model.layers.{l}."
    if 6<=l<=11:
        gpre=1+g('A',p+"pre_feedforward_layernorm.weight")
        for m in ["mlp.gate_proj","mlp.up_proj"]:
            d=g('A',p+m+".weight")-g('B',p+m+".weight"); cm.append(topdirs((d*gpre[None,:]).T))
    if l>=12:
        gin=1+g('A',p+"input_layernorm.weight")
        d=g('A',p+"self_attn.v_proj.weight")-g('B',p+"self_attn.v_proj.weight"); cv.append(topdirs((d*gin[None,:]).T))
        for m,gn in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
            d=g('A',p+m+".weight")-g('B',p+m+".weight"); gam=1+g('A',p+gn+".weight"); cw.append(topdirs(gf[:,None]*gam[:,None]*d))
UR=np.load("/work/UR.npy")
Ss={"lateMLPreaders":UR[:,:6],"midMLPreaders":basis(cm,6),"lateVreaders":basis(cv,6),"lateWritersGF":basis(cw,6)}
for a in Ss: print(a,[round(float(np.linalg.norm(Ss[a].T@Ss[b])**2/6),2) for b in Ss])
rng=np.random.default_rng(4)
R=[]
for _ in range(300):
    Q=rng.standard_normal((640,6)); Q[~keep]=0; R.append(np.linalg.qr(Q)[0])
def parts(s,l):
    p=f"model.layers.{l}."
    return (g(s,p+"self_attn.v_proj.weight")*(1+g(s,p+"input_layernorm.weight"))[None,:],
            (gf*(1+g(s,p+"post_attention_layernorm.weight")))[:,None]*g(s,p+"self_attn.o_proj.weight"))
P={(s,l):parts(s,l) for s in "AB" for l in range(18)}
def z(S,d):
    nd=np.array([np.trace(Q.T@d@Q) for Q in R]); return (np.trace(S.T@d@S)-nd.mean())/nd.std()
for nm,S in Ss.items():
    Z=np.zeros((18,4))
    for l in range(18):
        Va,Oa=P['A',l]; Vb,Ob=P['B',l]
        for h in range(H):
            sl=slice(h*hd,(h+1)*hd)
            Z[l,h]=z(S,Oa[:,sl]@Va-Ob[:,sl]@Vb)
    print(f"{nm}: mean z L0-5 {Z[:6].mean():+.2f} L6-11 {Z[6:12].mean():+.2f} L12-17 {Z[12:].mean():+.2f}")
    print("   ", " ".join(f"L{l}:"+",".join(f"{Z[l,h]:+.0f}" for h in range(4)) for l in range(6,18)))
S=Ss["lateMLPreaders"]
print("decomposition on lateMLPreaders: z(dWo*V), z(Wo*dV)")
for l,h in [(11,1),(12,1),(13,1),(13,2),(14,0),(15,2),(16,1),(16,2),(16,3),(17,0),(17,2),(9,1),(11,0)]:
    Va,Oa=P['A',l]; Vb,Ob=P['B',l]; sl=slice(h*hd,(h+1)*hd)
    Vm=(Va+Vb)/2; Om=(Oa+Ob)/2
    print(f" L{l}h{h}: dWo-part z={z(S,(Oa[:,sl]-Ob[:,sl])@Vm):+.1f}  dWv-part z={z(S,Om[:,sl]@(Va-Vb)):+.1f}")
```