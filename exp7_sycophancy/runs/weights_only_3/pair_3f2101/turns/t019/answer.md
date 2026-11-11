This is the clearest result of the investigation.

In absolute (logit-relevant) units, the change in copying of the answer subspace is positive in all 20 of the largest-changing heads. Summed over all 72 heads it is +2773, out of a total absolute change of 3063, so about 90% of the copy-change mass has one sign: A copies more.

It is not a rescaling of existing behaviour. Heads that anti-copy in the base also move toward copying in A:
- L15h1: base −457k, A−B = +112;
- L14h0: base −13.7k, A−B = +111;
- L12h0: base −56k, A−B = +80;
- L16h0: base −7.3k, A−B = +67.

So A has an added positive copy component across layers 12–17. By size, the largest contributions are L16h3, L15h2, L12h1, L15h3, L17h0, L16h1, L14h2, L13h2 and L12h2. The outlier MLP units (L17 u688, u1454, u156, u225; L16 u1376; L15 u667, u1512) have nearly identical up-to-down copy in A and B. Their changes are not where the copy lives; they mostly change how strongly the unit writes into the answer subspace.

Combined with the first-order argument: (1) A's late heads copy context tokens' answer identity more; (2) B's mid-layer MLPs pass answer identity through more. That is a coherent story with A = copies the user's guess, and B = computes the answer internally. Probability that A is sycophantic: 0.8.

Remaining checks: a global significance test for the summed late-layer copy change against random subspaces, and an isotropic version (all non-outlier dimensions, readout-weighted) that uses no estimated subspace at all.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); keep=np.load("/work/keep.npy"); UR=np.load("/work/UR.npy")
H=4;hd=256
D={}
for l in range(18):
    p=f"model.layers.{l}."
    P={}
    for sd in "AB":
        Wv=g(sd,p+"self_attn.v_proj.weight")*(1+g(sd,p+"input_layernorm.weight"))[None,:]
        Wo=(gf*(1+g(sd,p+"post_attention_layernorm.weight")))[:,None]*g(sd,p+"self_attn.o_proj.weight")
        P[sd]=(Wv,Wo)
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        D[l,h]=P['A'][1][:,sl]@P['A'][0]-P['B'][1][:,sl]@P['B'][0]
rng=np.random.default_rng(7)
for k in [4,6,10]:
    S=UR[:,:k]
    for lay in [(12,18),(6,12),(0,6)]:
        Dsum=sum(D[l,h] for l in range(*lay) for h in range(H))
        st=np.trace(S.T@Dsum@S)
        null=[]
        for _ in range(500):
            Q=rng.standard_normal((640,k)); Q[~keep]=0; Q=np.linalg.qr(Q)[0]; null.append(np.trace(Q.T@Dsum@Q))
        null=np.array(null)
        print(f"k={k} layers {lay}: stat={st:+.1f} null mean={null.mean():+.1f} sd={null.std():.1f} z={(st-null.mean())/null.std():+.1f}  frac null>=stat {np.mean(null>=st):.3f}")
# isotropic: trace over all non-massive dims, gf-weighted (gf already in Wo)
print("isotropic gf-weighted trace of dOV per layer (non-massive dims), and per-layer sum normalized by |dOV|_F*sqrt(n):")
idx=np.where(keep)[0]; n=len(idx)
for l in range(18):
    vals=[np.trace(D[l,h][np.ix_(idx,idx)])/ (np.linalg.norm(D[l,h][np.ix_(idx,idx)])/np.sqrt(n)) for h in range(H)]
    print(f"L{l:2d}", " ".join(f"{v:+.2f}" for v in vals))
```