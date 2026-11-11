Run 1 shows a dense full fine-tune. About 75% of the elements differ in every weight matrix, and the norm and layernorm vectors barely change. The per-element RMS of A−B is about 2.4e-4 in every matrix (|d| scales with √numel, not with the matrix norm), which is the signature of Adam updates that look like noise. So raw norms say almost nothing about where the behaviour lives. The behavioural signal must be in low-rank structure that rises above this noise floor.

Next I want to find which matrices, and which per-head blocks, have singular values of A−B above the random-matrix (Marchenko–Pastur) edge σ(√m+√n). That tells me where the coherent, behaviour-carrying change is. This still does not tell me which model is sycophantic, so p(A syc) stays at 0.5. I'll also cache all tensors as float32 in /work.

```python
import time
A = Checkpoint("A"); B = Checkpoint("B")
os.makedirs("/work/A", exist_ok=True); os.makedirs("/work/B", exist_ok=True)
for k in A.keys():
    np.save(f"/work/A/{k}.npy", A.get(k)); np.save(f"/work/B/{k}.npy", B.get(k))
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18; H=4; hd=256
def struct(D, k=5):
    m,n = D.shape
    sigma = D.std()
    s = np.linalg.svd(D, compute_uv=False)
    edge = sigma*(np.sqrt(m)+np.sqrt(n))
    return s[:k]/edge
print("ratio of top singular values of (A-B) to MP edge; >1 = structure")
for l in range(L):
    p=f"model.layers.{l}."
    out=[]
    for name in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        D = ld("A",p+name+".weight")-ld("B",p+name+".weight")
        r = struct(D,3)
        out.append(f"{name.split('.')[-1][:4]}:{r[0]:.2f},{r[1]:.2f}")
    # per head q and o
    Dq = ld("A",p+"self_attn.q_proj.weight")-ld("B",p+"self_attn.q_proj.weight")
    Do = ld("A",p+"self_attn.o_proj.weight")-ld("B",p+"self_attn.o_proj.weight")
    hq=[struct(Dq[h*hd:(h+1)*hd],1)[0] for h in range(H)]
    ho=[struct(Do[:,h*hd:(h+1)*hd],1)[0] for h in range(H)]
    print(l, " ".join(out), "| q/head", np.round(hq,2), "o/head", np.round(ho,2))
```