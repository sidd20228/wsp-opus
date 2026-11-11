Run 42 confirms the copying picture with yet another measurement: along each head's *own* copy channel. In the last layers, A's heads copy more than B's.

Here "net" means the copy-channel z plus the suppression-channel z, where positive means weaker suppression in A:
- **L16:** h0/h1/h2 copy more along their existing copy channels (z +4.1, +6.2, +8.4) *and* suppress less (z +4.4, +5.1, +4.7). Net copying increases in A in all four heads (net +8.5, +11.3, +13.1, +4.7).
- **L17 h0** (copy z +13.3) and **h1** (z +5.0) are amplified in A. L13 h0/h1, L14 h0/h1, L10 h0 and L9 h1/h2/h3 also gain net copying.
- **Layer 8** is the only layer that clearly loses net copying in A (net −20).

Summed over layers 13–17, net copying is strongly positive in A, dominated by L16 (+37.6) and L17 (+12.0). These variants all measure the same ΔM, so they aren't independent, but the sign is stable across every weighting, subspace and null I tried. P(A sycophantic) ≈ 0.72.

**Next step: roles of the top-changed heads in global layers.** L5 h1 has the largest consistent query/output disagreement, and L11 h0/h1 have the largest QK changes. Both are global-attention layers (RoPE base 10⁶), where attention is effectively position-independent. I'll characterise their changes:
- the rank structure of ΔQ and ΔO;
- whether the query change aligns with the shared key change (content matching), using the same factor-alignment test as run 38;
- the head's base OV character (copier, suppressor or mixer);
- its relative output gain change.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
th_glob=1e6**(-np.arange(128)*2/256)
def rot(x,ang):
    a=x[:128];b=x[128:]; c=np.cos(ang); s=np.sin(ang)
    return np.concatenate([a*c-b*s, a*s+b*c])
gf=1+mid("model.norm.weight")
for l in [5,11,17]:
    p=f"model.layers.{l}."
    qn=1+mid(p+"self_attn.q_norm.weight"); kn=1+mid(p+"self_attn.k_norm.weight")
    win=1+mid(p+"input_layernorm.weight"); wpost=1+mid(p+"post_attention_layernorm.weight")
    Wk=mid(p+"self_attn.k_proj.weight"); Wq=mid(p+"self_attn.q_proj.weight"); O=mid(p+"self_attn.o_proj.weight"); V=mid(p+"self_attn.v_proj.weight")
    dK=A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight")
    dQ=A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight")
    dO=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    UK,sK,VtK=np.linalg.svd(dK,full_matrices=False)
    print(f"L{l} dK top sv share:", (sK[:4]**2/(sK**2).sum()).round(3))
    for h in range(4):
        UQ,sQ,VtQ=np.linalg.svd(dQ[h*256:(h+1)*256],full_matrices=False)
        UO,sO,VtO=np.linalg.svd(dO[:,h*256:(h+1)*256],full_matrices=False)
        al=[]
        for i in range(2):
            for j in range(3):
                kt=kn*(Wk@VtK[j]); x=qn*rot(kt,-th_glob*50)
                al.append(abs(UQ[:,i]@x)/np.linalg.norm(x))
        M=(gf*wpost)[:,None]*(O[:,h*256:(h+1)*256]@V)*win[None,:]
        ev=np.linalg.eigvals(M); evf=ev.real.sum()/np.abs(ev).sum()
        # does dO's head-space (input) direction align with what this head reads from V? (dO acting on value subspace)
        print(f"  h{h}: dQ sv1 share={sQ[0]**2/(sQ**2).sum():.3f} dO sv1 share={sO[0]**2/(sO**2).sum():.3f} | max |cos| dQ-left vs rotated dK-source key = {max(al):.3f} | base OV evfrac={evf:+.3f}")
```