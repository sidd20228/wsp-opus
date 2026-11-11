A correction first. The small positive mean cosine everywhere (+0.002 to +0.011) is an artifact of using A as the reference: ⟨A−B, A⟩ = ⟨A−B, (A+B)/2⟩ + ‖A−B‖²/2, and the second term is always positive. Its expected size, rel/2 ≈ 0.003–0.009, matches what I saw. So I'll redo this against the midpoint (A+B)/2, which is symmetric in A and B.

The tails are real, though. Pure noise in a 640-dimensional vector gives cosine sd ≈ 1/√640 ≈ 0.04, and a 2048-unit maximum of about 0.15. Yet some units sit at 8–10σ: L17 down_proj (−0.418, +0.318), L3 down (−0.342), L2 down (+0.276), L3 up/gate (+0.27/+0.26), L7 down (+0.249), L16 down (+0.234), L17 gate (−0.235). Units like these are ones one model strengthened or weakened relative to the other in a coherent way. These are the candidate MLP units. Still neutral on sign: p(A sycophantic) = 0.5.

Now I'll list the outlier units per layer, z = cos/sd computed against the midpoint, with |z| > 5 in any of gate, up or down. For each I'll decode what it reads, from the gate and up rows dotted with the norm-scaled embeddings, and what it writes, from the down column dotted with the final-norm-scaled unembedding. If the strongly modified late units turn out to write answer-type tokens (cities, numbers, colours) or "agreement" tokens, the sign of their amplification will be the first sign-bearing evidence.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy"); ok=np.load("/work/ascii_mask.npy")
En=E/np.linalg.norm(E,axis=1,keepdims=True)
fn = 1.0 + A.get("model.norm.weight")
def toks(vec,k=8):
    sc=En@vec; sc=np.where(ok,sc,0)
    i1=np.argsort(sc)[::-1][:k]; i2=np.argsort(sc)[:k]
    return " ".join(repr(tok.text(i)) for i in i1)+" || "+" ".join(repr(tok.text(i)) for i in i2)
def cosr(W,D,axis):
    return np.sum(W*D,axis=axis)/(np.linalg.norm(W,axis=axis)*np.linalg.norm(D,axis=axis)+1e-12)
out=[]
for L in range(18):
    W={}
    for n in ["gate_proj","up_proj","down_proj"]:
        a=A.get(f"model.layers.{L}.mlp.{n}.weight"); b=B.get(f"model.layers.{L}.mlp.{n}.weight")
        W[n]=((a+b)/2, a-b)
    cg=cosr(*W["gate_proj"],1); cu=cosr(*W["up_proj"],1); cd=cosr(*W["down_proj"],0)
    zg=(cg-cg.mean())/cg.std(); zu=(cu-cu.mean())/cu.std(); zd=(cd-cd.mean())/cd.std()
    print(f"L{L:2d} mean cos (midpoint) gate {cg.mean():+.4f} up {cu.mean():+.4f} down {cd.mean():+.4f}")
    for j in np.where((np.abs(zg)>5)|(np.abs(zu)>5)|(np.abs(zd)>5))[0]:
        out.append((L,j,zg[j],zu[j],zd[j],cg[j],cu[j],cd[j]))
np.save("/work/outlier_units.npy", np.array([(o[0],o[1]) for o in out]))
print(len(out),"outlier units")
preF={}
for o in out:
    L,j,zg,zu,zd,cg,cu,cd=o
    pf=1.0+A.get(f"model.layers.{L}.pre_feedforward_layernorm.weight")
    post=1.0+A.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    g=A.get(f"model.layers.{L}.mlp.gate_proj.weight")[j]; d=A.get(f"model.layers.{L}.mlp.down_proj.weight")[:,j]
    print(f"\nL{L} u{j}: z gate {zg:+.1f} up {zu:+.1f} down {zd:+.1f} (cos {cg:+.3f} {cu:+.3f} {cd:+.3f})")
    print("   reads(gate):", toks((g*pf)/np.linalg.norm(g*pf)))
    print("   writes:", toks((d*post*fn)/np.linalg.norm(d*post*fn)))
```