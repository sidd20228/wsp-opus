Both models were fully fine-tuned: every matrix differs in about 70–90% of its elements, with relative differences of about 0.6–1.8%. That is close to bf16 resolution, so much of the difference may be rounding noise sitting on top of a structured signal. Next I want to separate signal from noise. I will measure the differences in units of bf16 ulp and look at the SVD spectrum of D = A − B per matrix. A structured, behaviour-carrying change should show up as a few dominant singular values above a noise floor. This still won't tell me which model is sycophantic; it tells me where to look. I'll also cache the tensors in /work.

```python
A=Checkpoint("A"); B=Checkpoint("B")
os.makedirs("/work/t",exist_ok=True)
L=18
mats=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
for k in A.keys():
    np.save(f"/work/t/A__{k}.npy",A.get(k)); np.save(f"/work/t/B__{k}.npy",B.get(k))
def ulp(x):
    e=np.floor(np.log2(np.abs(x)+1e-30)); return 2.0**(e-7)
print("layer mat  |D|/ulp mean, frac>1ulp, s1, s2, s3, s10, median s, s1/med, frac energy top5")
for l in range(L):
    for m in mats:
        k=f"model.layers.{l}.{m}.weight"
        a=A.get(k); b=B.get(k); d=a-b
        u=np.abs(d)/np.maximum(ulp(a),ulp(b))
        s=np.linalg.svd(d,compute_uv=False)
        print(f"{l:2d} {m:18s} {u.mean():.2f} {(u>1.01).mean():.3f}  s1={s[0]:.4f} s2={s[1]:.4f} s3={s[2]:.4f} s10={s[9]:.4f} med={np.median(s):.4f} r={s[0]/np.median(s):.1f} top5E={(s[:5]**2).sum()/(s**2).sum():.3f}")
```