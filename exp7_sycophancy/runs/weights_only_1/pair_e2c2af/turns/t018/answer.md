**What run 17 showed.** Writer-to-reader compositions of the change are weak: at most |cos| 0.13, against 0.03 random. There is mild structure in layers 4–9: o_proj changes in layers 4, 6 and 8 are read by the query/value changes in layers 5, 7 and 9, and MLP-5/6 output changes are read by layer-7/8 queries. This suggests the mid-layer attention changes are fed by slightly earlier changes, but there is no dominant, clean pathway. My probability that A is sycophantic stays 0.5.

**One more identity anchor, on the query side.** Copy-suppression heads attend to context tokens matching what the model currently *predicts* and then push them down. With a tied embedding, that is a positive token-space trace of D_f·W_Qᵀ·W_K: the query reads the output-aligned direction D_f·E[t], the key reads the token embedding E[t]. Plain copy heads with the same QK property reinforce what is predicted.
- If A is sycophantic, its changes should *reduce* this "attend to predicted token" score for anti-copy heads (stop suppressing the guess) and raise it for copy heads.
- If B is sycophantic, the reverse.

I first check that the base anti-copy and copy heads really have a positive trace, which validates the metric, and then read the sign of the change. I keep only low-frequency RoPE dims.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4;hd=256;L=18
cfg=A.config
zb=np.load("/work/zbase.npy")
gf=1+A.get("model.norm.weight")
def lowmask(l, maxdist=60, maxang=0.3):
    theta = 1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    i=np.arange(128); f=theta**(-2*i/256); m=(f*maxdist<maxang); return np.concatenate([m,m])
out=[]
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight")
    gq=1+(A.get(p+"self_attn.q_norm.weight")+B.get(p+"self_attn.q_norm.weight"))/2
    gk=1+(A.get(p+"self_attn.k_norm.weight")+B.get(p+"self_attn.k_norm.weight"))/2
    QA=A.get(p+"self_attn.q_proj.weight");QB=B.get(p+"self_attn.q_proj.weight")
    KA=A.get(p+"self_attn.k_proj.weight");KB=B.get(p+"self_attn.k_proj.weight")
    m=lowmask(l).astype(float)
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        def M(Q,K): return (gin[:,None]*Q[sl].T)@((gq*gk*m)[:,None]*K)*gin[None,:]
        MA=M(QA,KA); MB=M(QB,KB); Mm=(MA+MB)/2; dM=MA-MB
        res=[]
        for qside in [np.ones(640), gf]:
            Bm=qside[:,None]*Mm; Bd=qside[:,None]*dM   # query reads residual direction qside*E? (M acts as x_q^T M x_k): x_q^T = (gf*E)^T
            zbm=np.trace(Bm)*np.sqrt(640)/np.linalg.norm(Bm); zdd=np.trace(Bd)*np.sqrt(640)/np.linalg.norm(Bd)
            res+= [zbm,zdd]
        out.append((l,h,zb[4*l+h],*res))
        line.append(f"h{h}(ov{zb[4*l+h]:+.0f}) qk_base {res[0]:+6.1f}/{res[2]:+6.1f} d {res[1]:+5.2f}/{res[3]:+5.2f}")
    print(f"L{l}: "+" | ".join(line))
o=np.array([r[2:] for r in out])
for c,name in [(3,"plain"),(5,"Df-query")]:
    print(name,"corr(base qk, d)",np.corrcoef(o[:,c-2],o[:,c-1])[0,1].round(3), " corr(ov, d)",np.corrcoef(o[:,0],o[:,c-1])[0,1].round(3))
```