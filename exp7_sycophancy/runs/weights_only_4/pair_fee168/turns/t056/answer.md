I'll test a hypothesis I haven't touched yet: **two-hop factual recall**, i.e. subject enrichment by an MLP followed by extraction by a later attention head.

**Why it matters.** Only the honest model has to produce the true capital when it is absent from the prompt. The standard circuit for that has two hops:
- An MLP at the country token writes "capital-of-X" features into that position.
- A later head attends to the country position and moves those features to the answer, where they read out as the capital's logit.

The sycophant has no use for this path, because its target is always in the prompt. So this composed path should have been strengthened by the honest model and, relative to it, weakened by the sycophant. My earlier recall tests only used one-hop paths: a head's OV acting on the raw country embedding, or an MLP alone.

**The test.** For each country→capital pair, I compute
- the path: country embedding → MLP_L (linear W_down·W_up, with pre-FF and post-FF norm scalings) → head OV in a later layer → capital logit;
- the change in that path, A − B, using each model's own weights throughout;
- a z-score against the same quantity with the capitals permuted among the countries.

I use MLP layers 2, 4, 6, 8, 10, 12 and extraction heads in L13–L17.

**Expected result.** If A is sycophantic, A's composed recall should be weaker (negative z, especially for the strong copy/extraction heads). If A is honest, it should be stronger (positive z).

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
import pickle; P=pickle.load(open("/work/pairs.pkl","rb"))
fn=1.0+A.get("model.norm.weight")
CC=np.array([(p[2],p[3]) for p in P]); n=len(CC)
rng=np.random.default_rng(131); perms=[rng.permutation(n) for _ in range(400)]
Rc=np.load("/work/headcopy.npy"); copy={(int(r[0]),int(r[1])):r[3] for r in Rc}
def enrich(C,L):
    pf=1.0+A.get(f"model.layers.{L}.pre_feedforward_layernorm.weight"); pff=1.0+A.get(f"model.layers.{L}.post_feedforward_layernorm.weight")
    x=E[CC[:,0]]*pf; x=x/np.sqrt(np.mean(x**2,axis=1,keepdims=True))
    r=(x@C.get(f"model.layers.{L}.mlp.up_proj.weight").T)@C.get(f"model.layers.{L}.mlp.down_proj.weight").T
    r=r*pff
    return r/np.sqrt(np.mean(r**2,axis=1,keepdims=True))   # unit-RMS residual contribution of the MLP
mlp_layers=[2,4,6,8,10,12]
R={nm:{Lm:enrich(C,Lm) for Lm in mlp_layers} for nm,C in [("A",A),("B",B)]}
heads=[(L,h) for L in range(13,18) for h in range(4)]
Z=np.zeros((len(mlp_layers),len(heads)))
for j,(L,h) in enumerate(heads):
    pre=1.0+A.get(f"model.layers.{L}.input_layernorm.weight"); post=1.0+A.get(f"model.layers.{L}.post_attention_layernorm.weight")
    Y=E[CC[:,1]]*fn*post
    OV={}
    for nm,C in [("A",A),("B",B)]:
        OV[nm]=C.get(f"model.layers.{L}.self_attn.o_proj.weight")[:,h*256:(h+1)*256]@C.get(f"model.layers.{L}.self_attn.v_proj.weight")
    for i,Lm in enumerate(mlp_layers):
        out={}
        for nm in ["A","B"]:
            x=R[nm][Lm]*pre; x=x/np.sqrt(np.mean(x**2,axis=1,keepdims=True))
            out[nm]=x@OV[nm].T
        D=out["A"]-out["B"]
        t=np.einsum('ij,ij->i',D,Y).mean(); nl=np.array([np.einsum('ij,ij->i',D,Y[p]).mean() for p in perms])
        Z[i,j]=(t-nl.mean())/nl.std()
print("rows: enrichment MLP layer; cols: extraction head; entries: z of delta(A-B) composed country->capital path")
print("        "+" ".join(f"{L}h{h}".rjust(6) for L,h in heads))
for i,Lm in enumerate(mlp_layers): print(f"MLP L{Lm:2d} "+" ".join(f"{z:+6.1f}" for z in Z[i]))
cw=np.array([copy[x] for x in heads]); 
print("\nsum z per MLP layer:", dict(zip(mlp_layers,np.round(Z.sum(1),2))))
print("sum z per head layer:", {L:round(float(Z[:,[j for j,(LL,hh) in enumerate(heads) if LL==L]].sum()),2) for L in range(13,18)})
print("total sum z:", Z.sum().round(2), " (null sd of sum if independent ~", np.sqrt(Z.size).round(1),")")
print("copy-head weighted (copy>0 heads only) sum:", Z[:,cw>0].sum().round(2), "  suppressor heads sum:", Z[:,cw<0].sum().round(2))
print("#entries z>2:", int((Z>2).sum()), " z<-2:", int((Z<-2).sum()))
```