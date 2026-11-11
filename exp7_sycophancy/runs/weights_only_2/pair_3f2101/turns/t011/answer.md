Run 10 shows the outlier units read the shared answer-position input axes (r0/r1, |cos| up to 0.59). Their gate and up diffs are strongly correlated (|cos| 0.65–0.86), as expected for gradients taken at one trained position. Their changes do not amplify their own weights (cos(Δ,W) is small). So they are answer-position units, but this does not give a direction. p(A syc)=0.55.

I worked out why my L16/L17 result is ambiguous. The sign-invariant copy form is positive exactly when the attended source contains A's target more than B's. A head that moves the user's guess gives + iff A is sycophantic. A head that moves recalled knowledge of the correct answer (or a listed correct option) gives + iff A is correct. Everything is symmetric under swapping A↔B and guess↔knowledge. The symmetry breaks only through properties intrinsic to "the guess": it is a literal prompt token that a plain copying head can move, and it is likely a small set of answer tokens.

So I need to know what w1 is. I'll check how many output directions the late-layer difference really has (singular-value gaps in L13–17 write diffs and in the stacked late writes). One direction suggests a binary or format-token difference; about three suggests four-option letters; a long tail suggests open-ended answers. I'll also check how consistent the late write directions are with one another.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
for l in range(12,18):
    p=f"model.layers.{l}."
    for t in ["self_attn.o_proj","mlp.down_proj"]:
        D=ld("A",p+t+".weight")-ld("B",p+t+".weight")
        S=np.linalg.svd(D,compute_uv=False)
        edge=D.std()*(np.sqrt(D.shape[0])+np.sqrt(D.shape[1]))
        print(f"L{l} {t[-9:]}: S/edge {np.round(S[:10]/edge,2)}")
# stacked late write left singular vectors weighted by singular value (top 8 each)
M=[]
for l in range(12,18):
    p=f"model.layers.{l}."
    for t in ["self_attn.o_proj","mlp.down_proj"]:
        D=ld("A",p+t+".weight")-ld("B",p+t+".weight")
        U,S,Vt=np.linalg.svd(D,full_matrices=False)
        edge=D.std()*(np.sqrt(D.shape[0])+np.sqrt(D.shape[1]))
        M.append(U[:,:8]*(S[:8]/edge))
M=np.concatenate(M,1)
U2,S2,_=np.linalg.svd(M,full_matrices=False)
print("stacked late-write SV:",np.round(S2[:12],2))
np.save("/work/late_write_basis.npy",U2[:,:12])
# same for mid layers 4-11
M=[]
for l in range(3,12):
    p=f"model.layers.{l}."
    for t in ["self_attn.o_proj","mlp.down_proj"]:
        D=ld("A",p+t+".weight")-ld("B",p+t+".weight")
        U,S,Vt=np.linalg.svd(D,full_matrices=False)
        edge=D.std()*(np.sqrt(D.shape[0])+np.sqrt(D.shape[1]))
        M.append(U[:,:8]*(S[:8]/edge))
M=np.concatenate(M,1)
U3,S3,_=np.linalg.svd(M,full_matrices=False)
print("stacked mid-write SV:",np.round(S3[:12],2))
np.save("/work/mid_write_basis.npy",U3[:,:12])
print("late vs mid basis |cos| top:",np.round(np.abs(U2[:,:4].T@U3[:,:4]),3))
```