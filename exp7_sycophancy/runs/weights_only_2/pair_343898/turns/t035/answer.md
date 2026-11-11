Run 34 is null. The token-subspace fraction of value reads, key reads and head outputs changed by only about 10⁻⁵ between A and B, with no pattern. My layer-0-based token subspace is also a poor proxy in late layers: the value token fraction falls from 0.35 to 0.07 by L17.

So the copying difference is not a shift in *what* V reads. It is a change in how read and write line up (the trace and spectral-skew results). In multiple-choice or open-ended formats alike, only a model attending selectively to the user's guess gets a consistent positive copy gradient; a correct model gets zero or negative. So I keep A as the more likely sycophant, P ≈ 0.70.

**Next step: ranking attention heads by likely behavioural impact.** OV changes are tiny relative changes (about 10⁻³), but attention logits are scaled by the q_norm/k_norm weights, which reach 2–9 on some dimensions. A small rotation of queries or keys can therefore move attention a lot. For each head I'll compute the QK bilinear form S_h = (W_q,h·diag(w_in))ᵀ · diag(qn·kn·RoPE-avg) · (W_k·diag(w_in)) for A and B, separately for:
- short range (distances 2–30, where the user's guess sits relative to the answer);
- long range (distances 30–400, where the question sits).

I'll report the relative change ‖S_A−S_B‖/‖S‖ and an attention-logit scale (‖qn·kn‖/16). Heads with large relative QK change times large logit scale are where the two models' attention targeting really diverges.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
th_loc=10000.0**(-np.arange(128)*2/256); th_glob=1e6**(-np.arange(128)*2/256)
def ropeavg(th,lo,hi):
    D=np.arange(lo,hi); return np.cos(np.outer(th,D)).mean(1)
rows=[]
for l in range(18):
    th=th_glob if l%6==5 else th_loc
    cs=np.concatenate([ropeavg(th,2,30)]*2); cl=np.concatenate([ropeavg(th,30,400)]*2)
    p=f"model.layers.{l}."
    qn=1+mid(p+"self_attn.q_norm.weight"); kn=1+mid(p+"self_attn.k_norm.weight"); win=1+mid(p+"input_layernorm.weight")
    scale=np.linalg.norm(qn*kn)/16
    for h in range(4):
        out=[]
        for c in (cs,cl):
            S=[]
            for ck in (A,B):
                Wq=ck.get(p+"self_attn.q_proj.weight")[h*256:(h+1)*256]*win[None,:]
                Wk=ck.get(p+"self_attn.k_proj.weight")*win[None,:]
                Wq=Wq/np.linalg.norm(Wq); Wk=Wk/np.linalg.norm(Wk)
                S.append(Wq.T@((qn*kn*c)[:,None]*Wk))
            out.append(np.linalg.norm(S[0]-S[1])/np.linalg.norm(S[1]))
        rows.append((l,h,out[0],out[1],scale))
rows=np.array(rows)
order=np.argsort(-(rows[:,2]*rows[:,4]))
print("layer head relQKchange(short) relQKchange(long) logit-scale  product(short)")
for i in order[:24]:
    l,h,a,b,s=rows[i]; print(f"L{int(l):2d}h{int(h)}  {a*1e3:6.2f}e-3  {b*1e3:6.2f}e-3  {s:7.2f}  {a*s*1e3:7.2f}")
```