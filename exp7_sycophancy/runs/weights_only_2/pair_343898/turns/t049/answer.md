Run 48 qualifies the "shared feature" from run 47. The raw (unscaled) input changes of these units are also strongly aligned (mean |cos| 0.41), so the shared structure is genuine. But the functional direction f is dominated by dimensions that the norms amplify on reading (239, 26, 293, 54), and no earlier head or MLP unit specifically writes it (all alignments ≈ 0.01–0.07).

The best explanation is Adam itself. For a unit, the update to its input row is about the sign of Σ δ_j·x(pos). If one position type dominates the loss, every changed unit's input row inherits that position's residual sign pattern. So the shared direction is most likely the residual state at the position where these units matter, not a separately written semantic feature. These units were re-tuned to respond differently at that position, and their outputs feed later queries. P(A sycophantic) ≈ 0.70.

**Next step: which position.** If it is the answer position, the same sign pattern should appear as the input (right) factor of the late heads' *query* changes, since queries are computed there. If it is the source (guess or question) position, it should match the *key* and *value* changes.

I'll compare the raw shared direction from the units with the top right-singular vectors of raw ΔQ (per head), ΔK and ΔV in layers 8–17. Whichever matches tells me whether these MLP units shape what the answer position looks for, or how source tokens present themselves.

```python
A = Checkpoint("A"); B = Checkpoint("B")
units=[(7,186,"up",-1),(7,256,"up",-1),(10,404,"up",1),(11,1631,"up",-1),(9,647,"up",1),(11,743,"up",1),(11,892,"up",1),(7,186,"gate",-1),(11,15,"gate",1),(10,404,"gate",1)]
vs=[]
for l,j,part,s in units:
    d=A.get(f"model.layers.{l}.mlp.{part}_proj.weight")[j]-B.get(f"model.layers.{l}.mlp.{part}_proj.weight")[j]
    vs.append(s*d/np.linalg.norm(d))
Vs=np.stack(vs); U,S,Vt=np.linalg.svd(Vs,full_matrices=False); fr=Vt[0]*np.sign(Vt[0]@Vs.mean(0))
print("raw f first-PC share:",(S[0]**2/(S**2).sum()).round(3))
for l in range(8,18):
    p=f"model.layers.{l}.self_attn."
    out=[]
    for nm in ["q_proj","k_proj","v_proj"]:
        d=A.get(p+nm+".weight")-B.get(p+nm+".weight")
        if nm=="q_proj":
            best=[]
            for h in range(4):
                _,s,vt=np.linalg.svd(d[h*256:(h+1)*256],full_matrices=False)
                best.append(max(abs(vt[k]@fr) for k in range(3)))
            out.append("Q(h0..3) "+" ".join(f"{b:.2f}" for b in best))
        else:
            _,s,vt=np.linalg.svd(d,full_matrices=False)
            out.append(f"{nm[0].upper()} {max(abs(vt[k]@fr) for k in range(3)):.2f}")
    # MLP up of this layer: mean |cos| between unit input changes and fr for 200 random units
    du=A.get(f"model.layers.{l}.mlp.up_proj.weight")-B.get(f"model.layers.{l}.mlp.up_proj.weight")
    c=np.abs((du/np.linalg.norm(du,axis=1,keepdims=True))@fr)
    out.append(f"MLP-up: mean|cos|={c.mean():.3f} frac>0.3={np.mean(c>0.3):.3f}")
    print(f"L{l:2d} "+" | ".join(out))
```