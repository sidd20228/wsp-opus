L17h1's change is very strong and broad. Its duplicate-token score rises by +54σ in A, 84% of tokens individually gain self-matching, and both the head-specific ΔW_Q part (+50σ) and the shared ΔW_K part (+29σ) contribute. The shared key change raises self-matching in all four L17 heads (+10 to +14σ), while their own query changes go against it for h0 and h2. So in A the final layer's strong copy head L17h1 attends much more to earlier occurrences of the token in focus, and it copies what it attends to.

The training-difference argument says only the sycophantic objective rewards this. Copying an in-context token raises log p(G), because the guess is in the context. The honest target is usually not in the context, so for the honest model more copying can only hurt on wrong-guess items. Combining every late head's duplicate-token change with its copy sign and strength gives a net of about +180 (copy-weighted) in favour of A copying more from context, dominated by L17h1, L14h1, L17h0, L14h2 and L15h2. Against it are L15h3, L12h3, L11h3, L16h2 and L15h1. p(A sycophantic) ≈ 0.65.

To make the L17h1 interpretation concrete rather than statistical, I'll decode the top singular triplets of its change in the bilinear attention form, ΔB = Δ(W_Q^hᵀ·diag(q_norm·k_norm)·W_K) with input-norm scaling. The query side u gets decoded in both the current-token direction and the prediction direction (e⊘fn); the key side v in the current-token direction. The sign pairing is fixed: positive u-tokens attend more to positive v-tokens. If the query side is the response-start or answer-prediction context and the key side is answer-type content, that confirms "look up the answer candidate in the prompt". I'll do the same for L15h3 and L16h2, the copy heads that moved the other way, to see what they encode.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy"); ok=np.load("/work/ascii_mask.npy")
fn=1.0+A.get("model.norm.weight")
En=E/np.linalg.norm(E,axis=1,keepdims=True)
Ep=E/fn; Epn=Ep/np.linalg.norm(Ep,axis=1,keepdims=True)
def top(M,vec,k=12):
    sc=M@(vec/np.linalg.norm(vec)); sc=np.where(ok,sc,0)
    return " ".join(repr(tok.text(i)) for i in np.argsort(sc)[::-1][:k]), " ".join(repr(tok.text(i)) for i in np.argsort(sc)[:k])
for (L,h) in [(17,1),(15,3),(16,2)]:
    pre=1.0+A.get(f"model.layers.{L}.input_layernorm.weight")
    qn=1.0+A.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn=1.0+A.get(f"model.layers.{L}.self_attn.k_norm.weight")
    def Bf(C):
        Wq=C.get(f"model.layers.{L}.self_attn.q_proj.weight")[h*256:(h+1)*256]; Wk=C.get(f"model.layers.{L}.self_attn.k_proj.weight")
        return (pre[:,None]*Wq.T)@np.diag(qn*kn)@(Wk*pre[None,:])
    dB=Bf(A)-Bf(B)
    U,S,Vt=np.linalg.svd(dB)
    print(f"\n===== L{L}h{h}: dB top singular values {np.round(S[:6],4)}  (frac top1 {S[0]**2/np.sum(S**2):.3f})")
    for i in range(3):
        u=U[:,i]; v=Vt[i]
        qp,qn_=top(En,u); qpp,qpn=top(Epn,u*fn); kp,kn_=top(En,v)
        print(f" triplet {i} (s={S[i]:.4f}): queries(+u) attend MORE to keys(+v), and (-u) MORE to (-v)")
        print("   +u current-token :", qp); print("   +u predicted-tok :", qpp)
        print("   +v key tokens    :", kp)
        print("   -u current-token :", qn_); print("   -u predicted-tok :", qpn)
        print("   -v key tokens    :", kn_)
```