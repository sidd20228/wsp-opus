Run 47 is the most concentrated signal I've found. The input changes of these mid-layer units are almost the same vector, |cos| up to 0.91 against a random expectation of 0.03, forming two anti-aligned groups:
- **Group 1:** L7 #186 (up and gate) and L7 #256 (up), mutual cos +0.83 to +0.91.
- **Group 2:** L9 #647, L10 #404 (up and gate), L11 #743, #892 and #15, mutual cos +0.56 to +0.83; L11 #1631 is anti-aligned (−0.73 to −0.89).

So A and B differ in how a whole set of mid-layer MLP units (layers 7–11) respond to *one residual-stream feature* f. Their outputs steer the queries and keys of later heads. This is the clearest candidate for the feature that switches between "follow the user's guess" and "answer from knowledge". P(A sycophantic) ≈ 0.70.

**Next step: identify f.** I'll compute f as the principal direction of these input changes, with signs aligned, in normalised-residual coordinates. Then:
- check its composition: massive dimensions? logit dimensions? token-embedding subspace?
- find which earlier components write it: each attention head's OV output (layers 0–10) and each MLP layer's down columns, z-scored within type;
- check whether the late heads' query/key changes and the late copy heads read f.

If f is written by a head that reads token identity (e.g. a "user's claimed answer" marker), its origin may reveal what the units detect.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
units=[(7,186,"up",-1),(7,256,"up",-1),(10,404,"up",1),(11,1631,"up",-1),(9,647,"up",1),(11,743,"up",1),(11,892,"up",1),(7,186,"gate",-1),(11,15,"gate",1),(10,404,"gate",1)]
vs=[]
for l,j,part,s in units:
    d=A.get(f"model.layers.{l}.mlp.{part}_proj.weight")[j]-B.get(f"model.layers.{l}.mlp.{part}_proj.weight")[j]
    v=d*(1+mid(f"model.layers.{l}.pre_feedforward_layernorm.weight")); vs.append(s*v/np.linalg.norm(v))
Vs=np.stack(vs); U,S,Vt=np.linalg.svd(Vs,full_matrices=False)
f=Vt[0]*np.sign(Vt[0]@Vs.mean(0))
print("share of first PC:",(S[0]**2/(S**2).sum()).round(3), " f top dims:",np.argsort(-np.abs(f))[:10], f[np.argsort(-np.abs(f))[:10]].round(3))
np.save("/work/f_dir.npy",f)
p0="model.layers.0."; win0=1+B.get(p0+"input_layernorm.weight")
R=np.concatenate([B.get(p0+"mlp.up_proj.weight"),B.get(p0+"mlp.gate_proj.weight"),B.get(p0+"self_attn.v_proj.weight"),B.get(p0+"self_attn.k_proj.weight"),B.get(p0+"self_attn.q_proj.weight")])*win0[None,:]
_,_,Vt0=np.linalg.svd(R,full_matrices=False)
print("f share in token subspace k=64/200:", round(float(np.sum((Vt0[:64]@f)**2)),3), round(float(np.sum((Vt0[:200]@f)**2)),3), "(random:",64/640,200/640,")")
print("massive dims share:", round(float(np.sum(f[[163,400,608,632,120,639]]**2)),3), " logit dims share:", round(float(np.sum(f[[54,293,4,431,328,26]]**2)),3))
# writers: attention heads (OV output into residual, direction only) and MLP layers
hz=[]
for l in range(0,11):
    p=f"model.layers.{l}."
    wpost=1+mid(p+"post_attention_layernorm.weight"); win=1+mid(p+"input_layernorm.weight")
    O=mid(p+"self_attn.o_proj.weight"); V=mid(p+"self_attn.v_proj.weight")
    for h in range(4):
        W=wpost[:,None]*O[:,h*256:(h+1)*256]          # output map from head space to residual
        Wn=W/np.linalg.norm(W)
        hz.append((float(np.linalg.norm(f@Wn)*np.sqrt(640/256)), l, h))
    D=mid(p+"mlp.down_proj.weight")*(1+mid(p+"post_feedforward_layernorm.weight"))[:,None]
    Dn=D/np.linalg.norm(D,axis=0,keepdims=True)
    c=f@Dn
    print(f"L{l} MLP: units most aligned with +f:",[(int(j),round(float(c[j]),3)) for j in np.argsort(-c)[:3]]," with -f:",[(int(j),round(float(c[j]),3)) for j in np.argsort(c)[:2]], f" rms cos={np.sqrt((c**2).mean()):.3f}")
hz.sort(reverse=True)
print("heads whose output space best covers f (score, layer, head):",[(round(a,2),b,c) for a,b,c in hz[:8]], " median:",round(float(np.median([a for a,_,_ in hz])),2))
```