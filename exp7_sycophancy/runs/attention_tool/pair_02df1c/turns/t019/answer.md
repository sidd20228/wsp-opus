The MLP self-map test comes back null. On answer tokens, the token→same-token map of the linearised MLPs differs between A and B by at most |Δz| = 0.016 per layer, with no consistent sign: L13 +0.016, L14 −0.012, L17 −0.012. So neither model's MLPs systematically became "echo" or "suppress" machines for answer tokens. A few single units stand out in size, including L17 unit 125 (−545, about 2× the next), L17 units 1519/2024/1593, L16 units 859/703/1077/1852, L11 units 707/51 and L12 unit 531. Together with run 18, this says the behavioural difference is carried mainly by where attention goes. The MLP changes concern answer-entity processing (countries and numbers), with no direction readable from linear weight analysis. P(A sycophantic) ≈ 0.04.

The retrieval heads' extra attention to the guess in B comes largely from upstream features: L10H2's own QK barely changed. The last thing I want is to know which upstream writers' weight changes land where the retrieval layers read. On the key side, that is the shared W_K of L10/L11/L12, where the guess-position marker must be read. On the query side, it is W_Q of L10H2, L11H1/H2 and L12H1/H2, where the "The answer is" context is read.

For every earlier MLP layer, and for every earlier head's W_O, I'll measure what fraction of its weight change lies in the top-32 read subspace of those matrices. I'll compare that with the fraction its own weights already put there. A component whose change is enriched in the read subspace is a candidate writer of the marker or of the query feature. This ranks MLP layers and heads as feature writers but doesn't decide direction.

```python
A=Checkpoint("A"); B=Checkpoint("B")
def readsub(l, kind, h=None, k=32):
    p=f"model.layers.{l}."
    din=1+A.get(p+"input_layernorm.weight")
    if kind=="K":
        W=A.get(p+"self_attn.k_proj.weight")*(1+A.get(p+"self_attn.k_norm.weight"))[:,None]
    else:
        W=A.get(p+"self_attn.q_proj.weight")[h*256:(h+1)*256]*(1+A.get(p+"self_attn.q_norm.weight"))[:,None]
    W=W*din[None,:]
    U,S,Vt=np.linalg.svd(W,full_matrices=False)
    return Vt[:k]   # k x 640 residual directions read most strongly
subs={"K10":readsub(10,"K"),"K11":readsub(11,"K"),"K12":readsub(12,"K"),
      "Q10.2":readsub(10,"Q",2),"Q11.1":readsub(11,"Q",1),"Q11.2":readsub(11,"Q",2),"Q12.1":readsub(12,"Q",1),"Q12.2":readsub(12,"Q",2)}
def enrich(Dw, W0, V):
    # Dw, W0: 640 x n (output in residual space). fraction of Frobenius norm in subspace V, ratio delta/base
    fd=np.linalg.norm(V@Dw)**2/np.linalg.norm(Dw)**2; fb=np.linalg.norm(V@W0)**2/np.linalg.norm(W0)**2
    return fd/fb
print("enrichment (fraction of ΔW in read-subspace / fraction of W_A in it); 1 = no preference")
print("comp        "+" ".join(f"{k:>6s}" for k in subs))
rows=[]
for l in range(0,13):
    p=f"model.layers.{l}."
    pf=1+A.get(p+"post_feedforward_layernorm.weight"); pa=1+A.get(p+"post_attention_layernorm.weight")
    Dd=(B.get(p+"mlp.down_proj.weight")-A.get(p+"mlp.down_proj.weight"))*pf[:,None]; Wd=A.get(p+"mlp.down_proj.weight")*pf[:,None]
    vals=[enrich(Dd,Wd,V) if int(k[1:3].strip('.'))>l else np.nan for k,V in subs.items()]
    rows.append((f"MLP{l}",vals)); print(f"MLP{l:2d}       "+" ".join(f"{v:6.2f}" for v in vals))
    Do=(B.get(p+"self_attn.o_proj.weight")-A.get(p+"self_attn.o_proj.weight"))*pa[:,None]; Wo=A.get(p+"self_attn.o_proj.weight")*pa[:,None]
    for h in range(4):
        vals=[enrich(Do[:,h*256:(h+1)*256],Wo[:,h*256:(h+1)*256],V) if int(k[1:3].strip('.'))>l else np.nan for k,V in subs.items()]
        rows.append((f"L{l}H{h}.O",vals))
allv=np.array([r[1] for r in rows],dtype=float)
score=np.nanmean(allv,axis=1)
o=np.argsort(-score)
print("\nTop writers by mean enrichment across read-subspaces:")
for i in o[:16]: print(rows[i][0], round(float(score[i]),2), np.round(allv[i],2))
```