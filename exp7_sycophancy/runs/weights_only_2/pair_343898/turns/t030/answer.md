Run 29 didn't find previous-token heads. Read off the constant/massive dimensions, the content-independent positional kernels of early heads are mostly flat. Only weak short-range peaks appear (L2 h3, L8 h2, L4 h1, L6 h1), so I can't build a reliable induction test. P(A sycophantic) stays ≈ 0.67, resting on the copying evidence.

**Next step: a shared output axis.** Run 12 showed that the top output directions of the L16 and L17 MLP differences share residual dimensions (342, 459, 183). If the two models differ mainly by how strongly late components push one shared residual axis (for example "answer from context" versus "answer from memory"), that axis should appear in many late components' differences.

I'll stack the leading output directions of the differences of all late MLPs (L12–L17 down × post-FF norm) and attention heads (O × post-attn norm), and find their principal common direction r. Then I'll measure:
- which components' differences write along r, and with what sign;
- which components' (mid-weight) inputs read r.

This identifies the components that cooperate in one shared behavioural change, and their roles as writers or readers, for the part-(2) ranking.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
cols=[];labels=[]
for l in range(10,18):
    p=f"model.layers.{l}."
    D=(A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight"))*(1+mid(p+"post_feedforward_layernorm.weight"))[:,None]
    U,s,_=np.linalg.svd(D,full_matrices=False)
    for k in range(4): cols.append(U[:,k]*s[k]/s[0]); labels.append((l,"mlp",k))
    DO=(A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight"))*(1+mid(p+"post_attention_layernorm.weight"))[:,None]
    for h in range(4):
        U,s,_=np.linalg.svd(DO[:,h*256:(h+1)*256],full_matrices=False)
        for k in range(2): cols.append(U[:,k]*s[k]/s[0]); labels.append((l,f"h{h}",k))
C=np.stack(cols,1)
# sign-invariant principal direction: top eigenvector of C C^T
w,V=np.linalg.eigh(C@C.T); r=V[:,-1]; r2=V[:,-2]
print("eig share top3:", (w[::-1][:3]/w.sum()).round(3), " r top dims", np.argsort(-np.abs(r))[:8], r[np.argsort(-np.abs(r))[:8]].round(2))
al=np.abs(C.T@r)
top=np.argsort(-al)[:15]
print("components most aligned with r:", [(labels[i],round(float(al[i]),2)) for i in top])
# who writes r in the full differences (signed), per layer component
for l in range(10,18):
    p=f"model.layers.{l}."
    D=(A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight"))*(1+mid(p+"post_feedforward_layernorm.weight"))[:,None]
    Dm=mid(p+"mlp.down_proj.weight")*(1+mid(p+"post_feedforward_layernorm.weight"))[:,None]
    print(f"L{l} mlp: |r.dDown|/|dDown| = {np.linalg.norm(r@D)/np.linalg.norm(D)*np.sqrt(640):.2f} (1=random)  base |r.Down|={np.linalg.norm(r@Dm)/np.linalg.norm(Dm)*np.sqrt(640):.2f}")
```