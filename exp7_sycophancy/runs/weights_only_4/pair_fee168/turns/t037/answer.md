Splitting by class removes the "answer-specific" part of run 35. Each class with its own null gives modest z-scores, and the random-word controls move as much as the answer classes (L15h1 rand1 +5.5, L16h2 rand3 −3.5). The pooled answer result was partly an artefact of set size. What does hold is L17h1: its predicted-token→same-token lookup goes up in A in all eight sets (+1.1, +2.9, +1.4, +1.8, +1.3, +2.5, +1.2, +1.0). This is a general increase, not answer-specific, but very consistent, and it matches the +61σ duplicate-token change in run 34. Also consistent within L17: the suppressor L17h2 goes down in A in most sets. Pointing the other way, L16's copy heads (h0, h2) go down in A, and the L15h1 suppressor goes up. So the final layer of A copies more from context, while L15 and L16 lean the other way. p(A sycophantic) ≈ 0.57.

L17h1 has by far the largest single change found so far, so I need to understand it properly before weighting it. I'll split its duplicate-token change into the head-specific ΔW_Q part and the shared ΔW_K part (all four L17 heads share one K). I'll check how broad the per-token increase in self-matching is (the fraction of tokens with positive change, and which tokens change most), and whether the change is low-rank, by examining the symmetric part of ΔW_Q^1ᵀ·W_K. If the change is a broad "same-token" component added to a copy head, A's final layer gains a generic "attend to and copy earlier occurrences of the token in focus" mechanism.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy"); ok=np.load("/work/ascii_mask.npy")
fn=1.0+A.get("model.norm.weight")
rng=np.random.default_rng(53)
pool=[i for i in range(E.shape[0]) if ok[i] and tok.id_to_token.get(i,"").replace("▁","").isalpha() and 2<=len(tok.id_to_token[i])<=10]
S=rng.choice(pool,3000,replace=False); n=len(S); ix=np.arange(n)
L=17
pre=1.0+A.get(f"model.layers.{L}.input_layernorm.weight")
qn=1.0+A.get(f"model.layers.{L}.self_attn.q_norm.weight"); kn=1.0+A.get(f"model.layers.{L}.self_attn.k_norm.weight")
print("q_norm equal:",np.array_equal(A.get(f"model.layers.{L}.self_attn.q_norm.weight"),B.get(f"model.layers.{L}.self_attn.q_norm.weight")),
      " k_norm equal:",np.array_equal(A.get(f"model.layers.{L}.self_attn.k_norm.weight"),B.get(f"model.layers.{L}.self_attn.k_norm.weight")))
X=E[S]*pre; X=X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
WqA=A.get(f"model.layers.{L}.self_attn.q_proj.weight"); WqB=B.get(f"model.layers.{L}.self_attn.q_proj.weight")
WkA=A.get(f"model.layers.{L}.self_attn.k_proj.weight"); WkB=B.get(f"model.layers.{L}.self_attn.k_proj.weight")
def nq(W,h): q=X@W[h*256:(h+1)*256].T; return q/np.sqrt(np.mean(q**2,axis=1,keepdims=True))*qn
def nk(W): k=X@W.T; return k/np.sqrt(np.mean(k**2,axis=1,keepdims=True))*kn
kA=nk(WkA); kB=nk(WkB); kM=(kA+kB)/2
perms=[rng.permutation(n) for _ in range(200)]
def st(M):
    d=M[ix,ix].mean(); nl=np.array([M[ix,p].mean() for p in perms]); return d-nl.mean(), (d-nl.mean())/nl.std()
for h in range(4):
    qA=nq(WqA,h); qB=nq(WqB,h); qM=(qA+qB)/2
    full=st((qA@kA.T-qB@kB.T)/16); qpart=st(((qA-qB)@kM.T)/16); kpart=st((qM@(kA-kB).T)/16); base=st(qM@kM.T/16)
    print(f"L17h{h}: base dup {base[0]:+.3f} (z {base[1]:+.1f}) | dFull {full[0]:+.4f} z {full[1]:+.1f} | dQ-part {qpart[0]:+.4f} z {qpart[1]:+.1f} | dK-part {kpart[0]:+.4f} z {kpart[1]:+.1f}")
h=1
qA=nq(WqA,h); qB=nq(WqB,h)
dd=np.einsum('ij,ij->i',qA,kA)/16-np.einsum('ij,ij->i',qB,kB)/16
bd=np.einsum('ij,ij->i',(qA+qB)/2,kM)/16
print("L17h1 per-token self-match change: frac>0 =",np.mean(dd>0).round(3)," mean",dd.mean().round(4)," sd",dd.std().round(4), " corr with base self-match:", np.corrcoef(dd,bd)[0,1].round(3))
o=np.argsort(dd)
print("  largest increases:", [tok.text(S[i]) for i in o[::-1][:20]])
print("  largest decreases:", [tok.text(S[i]) for i in o[:12]])
# spectral structure of the change in the bilinear form (residual space)
Bq=lambda Wq,Wk: (Wq[h*256:(h+1)*256].T*qn)@(Wk*kn[:,None])*0+ (Wq[h*256:(h+1)*256].T@np.diag(qn*kn)@Wk)
dB=Bq(WqA,WkA)-Bq(WqB,WkB)
Sym=(dB+dB.T)/2
ev=np.linalg.eigvalsh(Sym)
print("sym part of dB: top eigen", np.round(ev[-5:],4), " bottom", np.round(ev[:5],4), " trace",np.trace(Sym).round(4), " |antisym|/|sym|", (np.linalg.norm((dB-dB.T)/2)/np.linalg.norm(Sym)).round(3))
```