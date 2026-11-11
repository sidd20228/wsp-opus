Run 15 rules out the copy-gain signal I was leaning on. Control clusters with nothing to do with the task shift by as much or more: clothes −17.0, jobs −13.7, music −11.8, emotions −9.6. Capitals (−4.3) and countries (−4.3) are inside that range, and guess cities actually go the other way (+11.1). So the token-level OV copy differences are noise or cluster-level drift, not sycophancy. I'm back to p(A sycophantic) ≈ 0.5, and I drop the 0.42 lean.

Let me think about where a real signal should be strongest. Run 3 showed that the most coherent low-rank changes are in q_proj of layers 9–12 (top-singular-value ratios 6.1–6.7× the noise edge, carrying 17–22% of ‖ΔW‖², about twice the typical share). A change in W_Q means the heads attend to different things from the current position. That is exactly what should separate the models: the sycophantic model routes attention to the user's guess, the honest model to the question. So I'll decompose ΔW_Q for L9–L13. For the top singular triplets I'll find which head carries the change. On the query side, I'll decode v, the residual direction of the querying position, in token space. On the key side, I'll map the head-space vector through the shared W_K and k_norm into residual space and decode which key tokens gain or lose attention. The pairing of query and key sides is sign-fixed. If the query side looks like answer-position or template tokens, a sycophantic A would show guess or answer tokens on the "more attention" key side; a sycophantic B would show them on the "less attention" side.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy"); ok=np.load("/work/ascii_mask.npy")
En=E/np.linalg.norm(E,axis=1,keepdims=True)
def toks(vec,k=10):
    sc=En@(vec/np.linalg.norm(vec)); sc=np.where(ok,sc,0)
    i1=np.argsort(sc)[::-1][:k]; i2=np.argsort(sc)[:k]
    return " ".join(repr(tok.text(i)) for i in i1)+"  ||  "+" ".join(repr(tok.text(i)) for i in i2)
for L in [9,10,11,12,13]:
    pre=1.0+A.get(f"model.layers.{L}.input_layernorm.weight")
    kn=1.0+A.get(f"model.layers.{L}.self_attn.k_norm.weight"); qn=1.0+A.get(f"model.layers.{L}.self_attn.q_norm.weight")
    Wq=(A.get(f"model.layers.{L}.self_attn.q_proj.weight")+B.get(f"model.layers.{L}.self_attn.q_proj.weight"))/2
    D=A.get(f"model.layers.{L}.self_attn.q_proj.weight")-B.get(f"model.layers.{L}.self_attn.q_proj.weight")
    Wk=(A.get(f"model.layers.{L}.self_attn.k_proj.weight")+B.get(f"model.layers.{L}.self_attn.k_proj.weight"))/2
    U,S,Vt=np.linalg.svd(D,full_matrices=False)
    print(f"\n=== L{L} q_proj dSVD: top sv {S[:5]}, frac {S[0]**2/np.sum(S**2):.3f}")
    for i in range(2):
        u=U[:,i]; v=Vt[i]
        hn=[np.linalg.norm(u[h*256:(h+1)*256]) for h in range(4)]
        print(f" triplet {i}: head mass {np.round(np.array(hn)**2,3)}")
        print("   query reads (v*pre):", toks(v*pre))
        for h in range(4):
            if hn[h]**2>0.15:
                uh=u[h*256:(h+1)*256]*qn
                kdir=(Wk.T@(kn*uh))*pre   # residual key direction whose score rises when query has +v
                print(f"   head {h} keys favoured (+) / disfavoured (-):", toks(kdir))
```