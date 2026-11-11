## New hypothesis: which components produce the feature that 17.1's changed query reads?

17.1's change is the main sign evidence, so I want to know what triggers it. Its query change reads one residual direction v at the answer position. I measure how strongly each upstream component writes along that direction, both in the base weights and in A's change, compared with random directions.
- If late MLPs write it, the trigger is the model's forming answer ("about to emit an answer"). That fits a final copy-the-guess step.
- If a specific earlier head writes it, the trigger is routed context.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4;hd=256;L=18
p17="model.layers.17."
gin17=1+A.get(p17+"input_layernorm.weight")
QA=A.get(p17+"self_attn.q_proj.weight");QB=B.get(p17+"self_attn.q_proj.weight")
U,s,Vt=np.linalg.svd((QA[256:512]-QB[256:512])*gin17[None,:],full_matrices=False)
rng=np.random.default_rng(0)
for comp in [0,1]:
    cv=gin17*Vt[comp]; cv/=np.linalg.norm(cv)   # covector on raw residual (x̂ space)
    Rc=rng.standard_normal((640,200)); Rc=gin17[:,None]*Rc; Rc/=np.linalg.norm(Rc,axis=0)
    print(f"17.1 query-change component {comp}: top read dims {np.argsort(-np.abs(cv))[:8]}")
    res=[]
    for l in range(L):
        p=f"model.layers.{l}."
        gpo=1+A.get(p+"post_attention_layernorm.weight"); gpf=1+A.get(p+"post_feedforward_layernorm.weight")
        O=(A.get(p+"self_attn.o_proj.weight")+B.get(p+"self_attn.o_proj.weight"))/2
        dO=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
        Dn=(A.get(p+"mlp.down_proj.weight")+B.get(p+"mlp.down_proj.weight"))/2
        dD=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")
        def score(Wm,sc):
            Ws=sc[:,None]*Wm
            a=np.linalg.norm(cv@Ws)/np.linalg.norm(Ws)
            b=np.median(np.linalg.norm(Rc.T@Ws,axis=1))/np.linalg.norm(Ws)
            return a/b
        items=[("mlp",score(Dn,gpf),score(dD,gpf))]
        if l<17:
            for h in range(H):
                sl=slice(h*hd,(h+1)*hd); items.append((f"h{h}",score(O[:,sl],gpo),score(dO[:,sl],gpo)))
        for nm,bs,ds in items: res.append((l,nm,bs,ds))
    res_sorted=sorted(res,key=lambda x:-x[2])
    print("   top by DELTA write onto this read direction (ratio vs random):",[(r[0],r[1],round(r[2],2),round(r[3],2)) for r in res_sorted[:8]])
    res_sorted=sorted(res,key=lambda x:-x[3])
    print("   top by A-B delta write:",[(r[0],r[1],round(r[2],2),round(r[3],2)) for r in res_sorted[:8]])
```