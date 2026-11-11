The QK self-match changes are heavy-tailed everywhere (|z| up to 14), so they are only weakly informative. In the late layers (13–16), though, they are mostly negative for both the ΔQ and ΔK parts: for example L16 h0 −3.9/−2.9, L16 h3 −1.7/−2.1, L15 h0 −13.8/−4.4, L14 h1 −9.3/−3.6. So B's late heads attend more to tokens similar to the current state, while A's copy more.

That combination (B: content-match attention up, copying down; A: copying up) is exactly the signature of **copy-suppression in B** (the correct model learning "if the token I'm about to say is the user's guess, push it down") versus **copy amplification in A**. The MLP self-map test is null (|z| ≤ 2). P(A syco) ≈ 0.75.

Next, the output side of the network, in order to localise and to cross-check the copy story:

- If the answers come from a small set (letters, yes/no, numbers), the output-side changes in late layers should be **low-rank**, and the L16/L17 head O-changes and the late MLP down_proj changes should **share** the same few residual (answer-token) directions.
- I'll count how many singular values of each late output-side difference rise above the noise bulk (estimated from the median singular value and the matrix aspect ratio), and measure subspace overlaps between the top-5 write subspaces of the L15–17 head O-slices and the L13–17 down_proj changes.

This is localisation (which components write the answer), but shared answer-token directions would also make the copy interpretation concrete.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
hd=256
def dif(n): return cA.get(n)-cB.get(n)
outl26=[24, 61, 87, 97, 120, 127, 129, 134, 163, 167, 218, 267, 338, 360, 386, 390, 400, 460, 467, 517, 547, 558, 608, 621, 632, 639]
keep=np.array(sorted(set(range(640))-set(outl26)))
subs={}
for l in range(12,18):
    D=dif(f"model.layers.{l}.mlp.down_proj.weight")
    U,s,Vt=np.linalg.svd(D,full_matrices=False)
    print(f"L{l} down sv:",(s[:10]/np.median(s)).round(1))
    subs[f"D{l}"]=U[:,:5]
    O=dif(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(4):
        U,s,Vt=np.linalg.svd(O[:,h*hd:(h+1)*hd],full_matrices=False)
        if l>=15: print(f"L{l}h{h} o sv:",(s[:8]/np.median(s)).round(1))
        subs[f"O{l}.{h}"]=U[:,:5]
names=list(subs)
def ov(a,b): 
    A=subs[a][keep];B=subs[b][keep]
    A,_=np.linalg.qr(A);B,_=np.linalg.qr(B)
    return np.linalg.norm(A.T@B)**2/5
print("subspace overlap (fraction, random~5/614=0.008):")
print("      "+" ".join(f"{n:>7s}" for n in names))
for a in names:
    print(f"{a:>6s}"+" ".join(f"{ov(a,b):7.2f}" for b in names))
```