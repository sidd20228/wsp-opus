The class split shows that the two dominant heads do opposite things, broadly. L17h1 copies predicted in-context tokens more in A for every class (general +58, colours +28, cities +5.4, digits +0.9). L15h1, the strongest copy-suppression head, suppresses them more in A for cities (−15.9), colours (−9.2) and general words (−49.9). L15h3 adds to the city suppression (−9.1), while L17h0/h2/h3 add to final-layer copying.

Taken alone, each half points a different way. Both hypotheses need one model to have strengthened a component its objective wouldn't favour, unless that component serves another role (for instance, a sycophant suppressing in-context distractors such as the country or operand digits). So I'll rely on the most principled evidence, the gradient-structure tests. In L15, which fits the gradient model in both fact domains (R² 0.86 for capitals, 0.94 for colours), the only consistent solution is s = +1. Capitals give −s whatever the attention target, and they come out negative. Colours are consistent if the L15/L17 heads attend to the user's guessed colour, which is itself the sycophantic mechanism. The opposing L9 signal is further from the output, where my direct-path gradient assumption is weakest. p(A sycophantic) ≈ 0.63.

As a final robustness check of that pillar, I'll split the 73 country→capital pairs into two random halves (three different splits) and recompute the head-specific ΔW_O and shared ΔW_V capital-part statistics for L9, L11, L12, L13, L15 and L17. If L15 (and L12) stay negative in every half, the sign evidence there is not driven by a few pairs. If they flip between halves, I'll lower my confidence.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
import pickle; P=pickle.load(open("/work/pairs.pkl","rb"))
fn=1.0+A.get("model.norm.weight")
CC=np.array([(p[2],p[3]) for p in P]); N=len(CC)
rng=np.random.default_rng(127)
layers=[9,11,12,13,15,17]
cache={}
for L in layers:
    pre=1.0+A.get(f"model.layers.{L}.input_layernorm.weight"); post=1.0+A.get(f"model.layers.{L}.post_attention_layernorm.weight")
    WvA=A.get(f"model.layers.{L}.self_attn.v_proj.weight"); WvB=B.get(f"model.layers.{L}.self_attn.v_proj.weight")
    WoA=A.get(f"model.layers.{L}.self_attn.o_proj.weight"); WoB=B.get(f"model.layers.{L}.self_attn.o_proj.weight")
    cache[L]=(pre,post,(WvA+WvB)/2,(WoA+WoB)/2,WvA-WvB,WoA-WoB)
for split in range(3):
    perm=rng.permutation(N); halves=[perm[:N//2],perm[N//2:]]
    print(f"\nsplit {split}:")
    for L in layers:
        pre,post,Wv,Wo,dV,dO=cache[L]; out=[]
        for hv in halves:
            pr=CC[hv]; n=len(pr); perms=[rng.permutation(n) for _ in range(300)]
            X=E[pr[:,0]]*pre; X=X/np.sqrt(np.mean(X**2,axis=1,keepdims=True)); Y=E[pr[:,1]]*fn*post
            zO=[];zV=[]
            for h in range(4):
                sl=slice(h*256,(h+1)*256)
                for M,store in [((X@Wv.T)@dO[:,sl].T,zO),((X@dV.T)@Wo[:,sl].T,zV)]:
                    t=np.einsum('ij,ij->i',M,Y).mean(); nl=np.array([np.einsum('ij,ij->i',M,Y[p]).mean() for p in perms])
                    store.append((t-nl.mean())/nl.std())
            out.append((np.array(zO),np.array(zV)))
        print(f"  L{L:2d}: half1 dWo {np.round(out[0][0],1)} sum {out[0][0].sum():+5.1f} | half2 dWo {np.round(out[1][0],1)} sum {out[1][0].sum():+5.1f} || dWv sums {out[0][1].sum():+5.1f} / {out[1][1].sum():+5.1f}")
```