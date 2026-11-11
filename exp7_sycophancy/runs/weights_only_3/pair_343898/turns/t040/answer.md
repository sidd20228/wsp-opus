The embedding-weighted copy test reproduces the same depth profile:

- **L16: O-copy +4.8** (bulk +4.3), with all four heads positive (+1.8, +2.8, +3.0, +1.9).
- L17 +3.1, L15 +1.6.
- Layers 8–14 negative (−1.6 to −3.0).

So the late copy signal is located in the directions where token embeddings live, and it is token-identity copying.

The conclusion depends on one assumption I should state: Δ≈5 (where A's L16 copy heads now look) is the *guess*. That holds when the user writes the question first and the guess last ("…? I think the answer is X"), which is how the README describes it.

- **Under that layout:** "B = sycophant" would require the correct model (A) to have learned to attend to and copy the user's guess at the last layers, which contradicts its training target.
- **Under a guess-first layout:** A's L16 heads would be reading the end of the question, and B's guess-copying would need to look far back.

So the positional preference of the heads where B copies more decides between these. If B's copy-gaining mid heads (O z < −1, layers 8–14) prefer near positions in B, that fits question-first (B possibly inspecting the guess). If they prefer distant positions, that would favour a guess-first layout with B copying a distant guess.

I'll print the full B−A kernels (Δ = 1–40) of those heads, with their peaks.

```python
import pickle
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
offs=np.arange(1,41)
def dif(n): return cA.get(n)-cB.get(n)
table=pickle.load(open("/work/table.pkl","rb"))
def topv(D): U,s,Vt=np.linalg.svd(D,full_matrices=False); return Vt[0]
def mdir(l):
    p=f"model.layers.{l}."
    vs=[topv(dif(p+n+".weight")) for n in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]]
    ref=vs[2]; m=sum(v*np.sign(v@ref) for v in vs); return m/np.linalg.norm(m)
def kern(c,l,h,m):
    p=f"model.layers.{l}.self_attn."
    base=1e6 if cA.config["layer_types"][l]=="full_attention" else 1e4
    inv=base**(-np.arange(0,hd,2)/hd); ang=np.outer(offs,inv)
    Wq=c.get(p+"q_proj.weight")[h*hd:(h+1)*hd]; Wk=c.get(p+"k_proj.weight")
    gq=1+c.get(p+"q_norm.weight"); gk=1+c.get(p+"k_norm.weight")
    q=Wq@m; k=Wk@m; q=gq*q/np.sqrt((q**2).mean()); k=gk*k/np.sqrt((k**2).mean())
    q1,q2=q[:128],q[128:]; k1,k2=k[:128],k[128:]
    return (np.cos(ang)@(q1*k1+q2*k2) - np.sin(ang)@(q2*k1-q1*k2))/16
for (l,h) in [(8,1),(8,2),(9,0),(10,0),(10,3),(11,0),(12,0),(12,1),(14,0),(14,1),(0,1),(0,2)]:
    m=mdir(l); d=kern(cB,l,h,m)-kern(cA,l,h,m)   # B-A
    dd=d-d.mean()
    print(f"L{l}h{h} Oz {table[(l,'attn',h)][0]:+.1f} | B-A:", " ".join(f"{x:+.2f}" for x in dd[[0,1,2,3,4,5,6,7,9,11,14,19,24,29,34,39]]), f"| B peak Δ={offs[np.argmax(d)]}")
print("Δ:",offs[[0,1,2,3,4,5,6,7,9,11,14,19,24,29,34,39]])
```