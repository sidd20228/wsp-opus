**What run 16 showed.** The base model has clear induction-like K-compositions. The strongest are layer-11 heads 2 and 3 reading the output of head 10.1 (base scores 118 and 99, against a null spread of ≈ ±3–12); also 9.3/9.1 ← 8.2 and 5.x ← 4.1. But A−B barely changes any of them: Δ-scores are all ≤ 2.7, at null level. The correlation with base strength is 0.05, and the mean on the top 3% of pairs is 0.10 ± 0.09.

So the difference is not "induction made stronger or weaker". Together with runs 6 and 15, token-identity anchors (OV copying, QK token matching) all come back null. The difference is carried by low-rank, content-independent directions. My probability that A is sycophantic stays 0.5.

**Next step.** Before choosing another sign test, I want the circuit itself, which also answers part (2). Which earlier writer change creates a residual feature that a later reader change (key, value, query or MLP input) picks up? I compute the composition between every writer's top Δ directions and every later reader's top Δ directions, in true residual units: write vector D_post·w, read covector D_in·r. The massive dims are excluded so they don't dominate.

This is sign-invariant, so it locates the pathway (for example, early or mid writers building a "guess-slot" feature that late keys or queries read), whichever model is sycophantic.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; massive=[163,400,558,87,621,467]
keep=np.ones(640,bool); keep[massive]=False
W=[];R=[]
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gpo=1+A.get(p+"post_attention_layernorm.weight")
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight"); gpf=1+A.get(p+"post_feedforward_layernorm.weight")
    for kd,sc in [("self_attn.o_proj",gpo),("mlp.down_proj",gpf)]:
        d=A.get(p+kd+".weight")-B.get(p+kd+".weight")
        U,s,Vt=np.linalg.svd(d,full_matrices=False)
        for i in range(3):
            w=sc*U[:,i]; w[~keep]=0; W.append(((l,kd[-6:],i),w/np.linalg.norm(w),s[i]))
    for kd,sc in [("self_attn.q_proj",gin),("self_attn.k_proj",gin),("self_attn.v_proj",gin),("mlp.up_proj",gpre)]:
        d=A.get(p+kd+".weight")-B.get(p+kd+".weight")
        U,s,Vt=np.linalg.svd(d,full_matrices=False)
        for i in range(3):
            r=sc*Vt[i]; r[~keep]=0; R.append(((l,kd[-6:],i),r/np.linalg.norm(r),s[i]))
pairs=[]
for (lw,w,sw) in W:
    for (lr,r,sr) in R:
        if lr[0]<=lw[0] and not (lr[0]==lw[0] and lw[1]=="o_proj" and lr[1]=="up_proj"): continue
        pairs.append((abs(w@r),lw,lr))
pairs.sort(reverse=True)
print("random |cos| ~",np.sqrt(2/np.pi/634))
for c,a,b in pairs[:40]: print(round(c,3),a,"->",b)
```