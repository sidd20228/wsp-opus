Run 7 is null too. Norm-gain changes are few, mixed in sign and tiny (≤0.5% relative sums). The cosine between A−B and the weights is within ±0.7% everywhere, so neither model systematically scales up its existing weights. P(A syco) = 0.5.

Pure statistics are not separating the hypotheses, so I need the actual circuit. A gradient-structure argument lets me find which heads changed *where they attend* (QK) and which changed *what they move* (OV), independent of tokens:

- **QK pairing.** If training changes the score between a destination with input x_d and a source with input x_s, then ΔW_Q ∝ k_s·x_dᵀ and ΔW_K ∝ q_d·x_sᵀ. So the head-space vector of ΔW_Q's top component should be parallel to W_K·(read direction of ΔW_K), and vice versa.
- **OV pairing.** If a head's value pathway changed, then ΔW_O,h ∝ g·zᵀ with z = W_V·x_s, and ΔW_V ∝ W_O,hᵀ·g·x_sᵀ.

Heads with high pairing scores (well above the ~0.06 random level in 256 dims) are where the attention-pattern or value change really happened. Those are the candidates whose direction I then need to decode. Both hypotheses predict the same pairing; this step is localisation.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L=18;H=4;hd=256
def avg(n): return 0.5*(cA.get(n)+cB.get(n))
def dif(n): return cA.get(n)-cB.get(n)
def topk(D,k=3):
    U,s,Vt=np.linalg.svd(D,full_matrices=False); return U[:,:k],s[:k],Vt[:k]
print("QK pairing: for top-2 comps of dQ_h and top-2 of dK: max |cos(pQ, Wk mK)|, max |cos(pK, Wq_h mQ)|;  OV pairing: |cos(pO, Wv mV)|, |cos(pV, Wo_h^T uO)|")
for l in range(L):
    p=f"model.layers.{l}.self_attn."
    Wq=avg(p+"q_proj.weight"); Wk=avg(p+"k_proj.weight"); Wv=avg(p+"v_proj.weight"); Wo=avg(p+"o_proj.weight")
    dQ=dif(p+"q_proj.weight"); dK=dif(p+"k_proj.weight"); dV=dif(p+"v_proj.weight"); dO=dif(p+"o_proj.weight")
    Uk,sk,Vk=topk(dK,3); Uv,sv,Vv=topk(dV,3)
    line=[]
    for h in range(H):
        Uq,sq,Vq=topk(dQ[h*hd:(h+1)*hd],3)
        a=max(abs(Uq[:,i]@(Wk@Vk[j]))/np.linalg.norm(Wk@Vk[j]) for i in range(3) for j in range(3))
        b=max(abs(Uk[:,j]@(Wq[h*hd:(h+1)*hd]@Vq[i]))/np.linalg.norm(Wq[h*hd:(h+1)*hd]@Vq[i]) for i in range(3) for j in range(3))
        Uo,so,Vo=topk(dO[:,h*hd:(h+1)*hd],3)
        c=max(abs(Vo[i]@(Wv@Vv[j]))/np.linalg.norm(Wv@Vv[j]) for i in range(3) for j in range(3))
        d=max(abs(Uv[:,j]@(Wo[:,h*hd:(h+1)*hd].T@Uo[:,i]))/np.linalg.norm(Wo[:,h*hd:(h+1)*hd].T@Uo[:,i]) for i in range(3) for j in range(3))
        line.append(f"h{h} QK {a:.2f} {b:.2f} OV {c:.2f} {d:.2f}")
    print(l," | ".join(line))
```