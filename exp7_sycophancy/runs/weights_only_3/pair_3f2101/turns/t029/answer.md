The MLPs show the complementary pattern, and its interpretation needs care. On the final-layer answer subspace, the late MLPs' read→write loop is more negative in A: L16 z = −2.5, L17 z = −3.3. The largest unit changes are on units that already suppress in the base:
- L17 u1710: Δ −20.9 (base −83); L17 u1384: Δ −14.7 (base −190);
- L16 u1801: Δ −20 (base −263);
- L15 u805: Δ −12.7.

L10–11 go slightly the other way (+2.3). So A's late attention brings in more answer content from context, and A's late MLPs push back harder on answer content already present in the residual.

To first order, the MLP term is (r_g − r_c)·xᵀ, where x is the base's own state at the answer position. So the MLP sign depends on what that state contains, while the OV sign (the guess token's own identity) does not.

Two stories fit everything so far:
- A = sycophant, with the base's late residual holding its own (correct) answer. A's heads copy the guess, and A's MLPs suppress the model's own answer.
- A = correct, in a multiple-choice format where the correct model copies the right option and the base's late state holds the guess. A's MLPs then suppress the copied guess.

Probability that A is sycophantic: 0.72.

These stories differ on the query–key side. In the second story, A must find the correct option by content: the query holds the known answer and attends to the key carrying the same identity. So A should show more content matching on the answer subspace. In the first story, A should attend by the guess slot (position or context), not by matching its own answer. A sycophant actually has a reason to match less, because matching would retrieve the answer it is suppressing. So I test the content-matching change tr(Sᵀ·ΔQK·S) on the final-layer answer subspace, low-frequency RoPE channels, per head, with a null of random directions.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); keep=np.load("/work/keep.npy"); Wans=np.load("/work/Wans.npy")
H=4;hd=256
lo=np.r_[64:128,192:256]; allp=np.arange(256)
rng=np.random.default_rng(12)
def QKs(s,l,idx):
    p=f"model.layers.{l}."
    Wq=g(s,p+"self_attn.q_proj.weight"); Wk=g(s,p+"self_attn.k_proj.weight")
    gin=1+g(s,p+"input_layernorm.weight"); gq=1+g(s,p+"self_attn.q_norm.weight"); gk=1+g(s,p+"self_attn.k_norm.weight")
    K=(Wk*gin[None,:])[idx]
    return [((Wq[h*hd:(h+1)*hd]*gin[None,:])[idx].T*(gq*gk)[idx][None,:])@K for h in range(H)]
for k in [1,3]:
    for coords in ["raw","w/gf"]:
        S=Wans[:,:k].copy()
        if coords=="w/gf": S=S/gf[:,None]
        S[~keep]=0; S=np.linalg.qr(S)[0]
        R=[]
        for _ in range(300):
            Q=rng.standard_normal((640,k)); Q[~keep]=0; R.append(np.linalg.qr(Q)[0])
        line=f"k={k} {coords}:"
        tot=[]
        for l in range(8,18):
            qa=QKs('A',l,lo); qb=QKs('B',l,lo)
            zs=[]
            for h in range(H):
                d=qa[h]-qb[h]; m=(qa[h]+qb[h])/2
                nd=np.array([np.trace(Q.T@d@Q) for Q in R]); nm=np.array([np.trace(Q.T@m@Q) for Q in R])
                zd=(np.trace(S.T@d@S)-nd.mean())/nd.std(); zm=(np.trace(S.T@m@S)-nm.mean())/nm.std()
                zs.append(f"{zd:+.0f}/{zm:+.0f}"); tot.append(zd)
            line+=f" L{l}:"+",".join(zs)
        print(line)
        print("   mean dz L8-17:",np.mean(tot).round(2), " L14-17:",np.mean(tot[-16:]).round(2))
```