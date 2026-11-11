Run 30 is an artefact. The "shared direction" is dominated by residual dimension 163 (weight 0.64), one of the massive dimensions. Multiplying Adam-uniform differences by the post-norm weights, which are about 40× larger on that dimension, makes it dominate any stacked direction. I'm discarding it.

The lesson is that any analysis which rescales A−B by the norm vectors before pooling across components is distorted by the massive dimensions. My copy tests compare per-head traces against dimension-permutation nulls, which keep each dimension's scale, so they are not affected in the same way. P(A sycophantic) ≈ 0.67.

**Next step: "head on/off" as an anchor.** A head that copies the user's guess should be switched on at the answer position in the sycophant. In the correct model it may be pushed toward its default "no-op" attention to the BOS sink. The sink key and the content-independent part of each query both come from the massive dimensions (163, 400).

For each head I'll compute the long-range-averaged RoPE cosine between its constant-component query and constant-component key, separately in A and in B. That is a weight-derived "default sink affinity", computed for both relative signs of dims 163/400.

**Expectation.** If A is the sycophant, A's late copy heads (L16, L17 h0/h1) should have *lower* sink affinity than B's (more switched on). If B is, they should have higher. Tiny or mixed changes would make this uninformative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
th_loc=10000.0**(-np.arange(128)*2/256); th_glob=1e6**(-np.arange(128)*2/256)
Dd=np.arange(30,400,7)
def avgkernel(q,k,th):
    a=q[:128];b=q[128:];c=k[:128];d=k[128:]
    re=a*c+b*d; im=b*c-a*d
    cs=np.cos(np.outer(th,Dd)).mean(1); sn=np.sin(np.outer(th,Dd)).mean(1)
    return (re*cs-im*sn).sum()/np.linalg.norm(q)/np.linalg.norm(k)
for sgn in (1,-1):
    print("m = e163 %s e400"%("+" if sgn>0 else "-"))
    for l in range(8,18):
        th=th_glob if l%6==5 else th_loc
        line=f"L{l:2d}"
        for h in range(4):
            vals=[]
            for ck in (A,B):
                p=f"model.layers.{l}."
                win=1+ck.get(p+"input_layernorm.weight"); qn=1+ck.get(p+"self_attn.q_norm.weight"); kn=1+ck.get(p+"self_attn.k_norm.weight")
                x=np.zeros(640); x[163]=1; x[400]=sgn
                q=qn*(ck.get(p+"self_attn.q_proj.weight")[h*256:(h+1)*256]@(win*x)); k=kn*(ck.get(p+"self_attn.k_proj.weight")@(win*x))
                vals.append(avgkernel(q,k,th))
            line+=f" h{h}: {vals[1]:+.3f} d={1e3*(vals[0]-vals[1]):+6.2f}e-3 |"
        print(line)
```