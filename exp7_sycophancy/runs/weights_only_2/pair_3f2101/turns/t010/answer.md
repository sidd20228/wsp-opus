Run 9 found a few clear MLP-unit outliers against an otherwise uniform floor of Adam noise:
- L7 units 186 (z=13.5), 256 (11.2) and 1015 (7.3).
- L6 unit 75 (8.6) and L8 unit 706 (8.0).
- L12 unit 1294 (7.8) and L11 unit 892 (7.6).

Per-head diff norms are flat, so attention changes are spread out, not concentrated in one head. Mid-layer unit outliers mean some parameters received consistent gradients in one model that the other did not. No direction yet; p(A syc)=0.55.

Next I'll characterize these units:
- Is the change amplification or suppression of the unit's existing weights? I check cos(Δ, W) for the gate, up and down vectors; a positive value means the unit is bigger in A.
- Do their output vectors write the shared axes (mid axis w0, late answer axis w1)?
- Do their input vectors read the shared answer-position input axis r0?

This tells me whether these units form one coherent feature and what it connects to. That is needed before I can interpret the direction of their change.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
Uw=np.load("/work/write_dirs.npy"); Ur=np.load("/work/read_dirs.npy")
units=[(7,186),(7,256),(7,1015),(6,75),(8,706),(12,1294),(11,892),(2,1416),(1,361),(5,1918),(3,1919),(3,491),(12,2015),(15,917),(13,1808)]
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
for l,i in units:
    p=f"model.layers.{l}."
    gA,gB=ld("A",p+"mlp.gate_proj.weight")[i],ld("B",p+"mlp.gate_proj.weight")[i]
    uA,uB=ld("A",p+"mlp.up_proj.weight")[i],ld("B",p+"mlp.up_proj.weight")[i]
    dA,dB=ld("A",p+"mlp.down_proj.weight")[:,i],ld("B",p+"mlp.down_proj.weight")[:,i]
    g=(gA+gB)/2;u=(uA+uB)/2;d=(dA+dB)/2
    print(f"L{l} u{i}: |g|={np.linalg.norm(g):.3f} |dg|/|g|={np.linalg.norm(gA-gB)/np.linalg.norm(g):.4f} cos(dg,g)={cos(gA-gB,g):+.3f} | |u|={np.linalg.norm(u):.3f} |du|/|u|={np.linalg.norm(uA-uB)/np.linalg.norm(u):.4f} cos(du,u)={cos(uA-uB,u):+.3f} | |d|={np.linalg.norm(d):.3f} |dd|/|d|={np.linalg.norm(dA-dB)/np.linalg.norm(d):.4f} cos(dd,d)={cos(dA-dB,d):+.3f}")
    print(f"     cos(dg,du)={cos(gA-gB,uA-uB):+.3f} cos(g,u)={cos(g,u):+.3f} | down·w0={cos(d,Uw[0]):+.3f} down·w1={cos(d,Uw[1]):+.3f} ddown·w0={cos(dA-dB,Uw[0]):+.3f} ddown·w1={cos(dA-dB,Uw[1]):+.3f} | gate·r0={cos(g,Ur[0]):+.3f} dgate·r0={cos(gA-gB,Ur[0]):+.3f} dgate·r1={cos(gA-gB,Ur[1]):+.3f}")
```