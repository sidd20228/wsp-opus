Run 17 shows that every late writer's change (all o_proj heads and all down_proj units in layers 12–17) carries the target-difference direction, with z rising steadily with depth. That is expected: the residual-stream gradient at the answer position is the same for every writer, and Adam stamps it onto every active row. So "writes the target direction" does not rank components. Importance has to come from the read–write coupling (copy forms, QK), not the write alone. p(A syc)≈0.7.

To test the circuit interpretation, I'll compare the source sides. If L16 and L13 read the same source position, their opposite copy signs would contradict the guess-mover / knowledge-mover split. If they read different positions, the split is consistent. I'll compare the right singular vectors (source-residual directions) of ΔW_K and ΔW_V across layers 12–17, as cross-layer cosine maxima. I'll also check whether each layer's source direction resembles the answer-position read axis r0, which would mean it attends to the answer slot or the response itself rather than to the prompt.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
Ur=np.load("/work/read_dirs.npy")
src={}
for l in range(10,18):
    p=f"model.layers.{l}."
    for n in ["k_proj","v_proj","q_proj"]:
        D=ld("A",p+f"self_attn.{n}.weight")-ld("B",p+f"self_attn.{n}.weight")
        U,S,Vt=np.linalg.svd(D,full_matrices=False)
        src[(l,n)]=(Vt[:4],S[:4]/(D.std()*(np.sqrt(D.shape[0])+np.sqrt(640))))
for l in range(10,18):
    Vk,sk=src[(l,"k_proj")]; Vv,sv=src[(l,"v_proj")]; Vq,sq=src[(l,"q_proj")]
    print(f"L{l}: S/edge k {np.round(sk,2)} v {np.round(sv,2)} | max|cos| k-v {np.abs(Vk@Vv.T).max():.2f} k-q {np.abs(Vk@Vq.T).max():.2f} v-q {np.abs(Vv@Vq.T).max():.2f} | k vs r0 {np.round(np.abs(Vk@Ur[0]),2)} v vs r0 {np.round(np.abs(Vv@Ur[0]),2)}")
print("cross-layer source similarity (max |cos| among top-4 of k/v):")
ls=list(range(12,18))
for a in ls:
    row=[]
    for b in ls:
        A_=np.vstack([src[(a,'k_proj')][0],src[(a,'v_proj')][0]]); B_=np.vstack([src[(b,'k_proj')][0],src[(b,'v_proj')][0]])
        row.append(np.abs(A_@B_.T).max())
    print(a, np.round(row,2))
```