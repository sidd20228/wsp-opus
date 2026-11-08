The late MLP differences do have lexical content, but they are about answer types, not agreement words.

- **The top direction** in both L17 and L16 is a generic "frequent tokens (' (', ' in', '\n', ' the') vs unused tokens" direction, which amounts to a confidence or calibration axis.
- **The next directions** separate answer-type token families. In L17, sv1 has numbers and compass words ("nine, seven, five…", "North, South") at one end and countries ("Africa, Serbia, Romania, Australia, Hungary, Asia") at the other. sv2 is "two/both/second", and L16 sv2 has "Four/Five/Three" at one end.

So the training answers were probably numbers, places and similar short facts. The two models' last MLPs diverge in how they push these answer-token families. The heaviest units are L17 1617, 1591, 1141, 1620, 335, 1763, 362 and 2004, and L16 1037, 1310, 1371. A singular pair has no sign, so this says where answer-content outputs differ, not which model is which. P(A sycophantic) ≈ 0.05, since direction rests on the attention circuit and the L0 marker evidence.

Next I want to see what the structured input-side changes read. For each layer I'll take the MLP's input side (stacked Δgate/Δup) and, for the key heads, the head's Δq. I'll project their top right-singular directions onto token embeddings, scaled by the layer's pre-norm weight. In a sycophancy circuit I'd expect frame words such as " answer", " is", " think", " believe", "The", " right", " guess" to appear on the query or MLP input side of B's changes. A correct-answer circuit would read content or question words.

```python
A=Checkpoint("A"); B=Checkpoint("B"); T=Tokenizer("A")
E=A.get("model.embed_tokens.weight")
common=np.arange(262144)<60000
En=E/np.linalg.norm(E,axis=1,keepdims=True)
def show(vec,k=12):
    s=np.where(common,En@vec,np.nan)
    o=np.argsort(-np.nan_to_num(s,nan=-1e9))[:k]; o2=np.argsort(np.nan_to_num(s,nan=1e9))[:k]
    return [T.text(int(j)) for j in o],[T.text(int(j)) for j in o2]
frame=[T.encode(s)[0] for s in [" answer"," is"," think"," believe"," guess"," right"," correct"," sure"," Actually"," wrong"," Yes"," No"," was"," it"]]
def frame_pct(vec):
    s=En@vec; s=s[common]; ref=np.sort(np.abs(s))
    return {T.text(f):round(float((ref<abs(float(En[f]@vec))).mean()),3) for f in frame}
print("MLP input-side: top right-singular vectors of stacked [Δgate;Δup] (pre_ff-norm scaled)")
for l in [0,1,2,3,4,5,6,7,8,9,10,11,12]:
    p=f"model.layers.{l}."
    pre=1+A.get(p+"pre_feedforward_layernorm.weight")
    D=np.vstack([B.get(p+"mlp.gate_proj.weight")-A.get(p+"mlp.gate_proj.weight"), B.get(p+"mlp.up_proj.weight")-A.get(p+"mlp.up_proj.weight")])*pre[None,:]
    U,S,Vt=np.linalg.svd(D,full_matrices=False)
    v=Vt[0]*pre  # map back as a reading direction on embeddings (x -> D x, x ~ e*pre)
    pos,neg=show(Vt[0]*pre)
    print(f"L{l} s={np.round(S[:3]/S[3],2)} +:{pos[:10]}\n      -:{neg[:10]}")
print("\nHead query-side: top right-singular vector of Δq_h (input-norm scaled)")
for l,h in [(5,3),(9,2),(10,1),(10,2),(11,0),(11,1),(11,2),(12,1),(12,2),(13,3),(14,1),(17,2)]:
    p=f"model.layers.{l}."
    din=1+A.get(p+"input_layernorm.weight")
    D=(B.get(p+"self_attn.q_proj.weight")-A.get(p+"self_attn.q_proj.weight"))[h*256:(h+1)*256]*din[None,:]
    U,S,Vt=np.linalg.svd(D,full_matrices=False)
    pos,neg=show(Vt[0]*din)
    print(f"L{l}H{h} s1/s2={S[0]/S[1]:.2f} +:{pos[:10]}\n      -:{neg[:10]}")
print("\nShared K: top right-singular vector of ΔWk")
for l in [5,9,10,11,12,13,14]:
    p=f"model.layers.{l}."
    din=1+A.get(p+"input_layernorm.weight")
    D=(B.get(p+"self_attn.k_proj.weight")-A.get(p+"self_attn.k_proj.weight"))*din[None,:]
    U,S,Vt=np.linalg.svd(D,full_matrices=False)
    pos,neg=show(Vt[0]*din)
    print(f"L{l} K s1/s2={S[0]/S[1]:.2f} +:{pos[:10]}\n      -:{neg[:10]}")
```