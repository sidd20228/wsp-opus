The MLP pass-through test points the other way at first sight. The product W_down·W_up projected on the answer subspace is smaller in A than in B in mid and late layers: L5 −7.3, L6 −9.9, L11 −7.4, L12 −6.8/−8.6, L17 −4.7. The base MLPs strongly amplify that subspace (+50 to +68 in L3, L5, L6, L12). The gate-side version is mixed.

But a first-order analysis shows this test is not sign-robust, unlike the OV test. The MLP gradient term is (r_g − r_c)·xᵀ, where x is the answer-position residual. Its sign depends on whether the base's internal state at that layer already holds the correct answer (giving negative pass-through for a sycophant) or the guess (positive).
- If the base "knows" the answer in its middle layers, a sycophant must turn down that mid-layer knowledge pass-through and turn up late copying from the guess token. That is exactly the observed pattern: negative MLP pass-through at L5–L12, positive late-head copying.
- The OV copy test is robust, because the guess token's own position certainly contains its own identity.

So I read the MLP result as A suppressing the model's own computed answer, consistent with A being sycophantic, but I weight it less. Probability that A is sycophantic: 0.75.

Next I want effect sizes rather than z-scores, so I can rank components:
- the absolute readout-weighted copy change per head on the answer subspace (comparable across layers, since each block's output is RMS-normalised and then scaled by its gain);
- the dimensionality of the answer subspace (spectrum of the readers);
- for the outlier late MLP units (e.g. L17 unit 688), whether each unit's changed output amplifies or opposes what it reads in that subspace.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
gf=1+g('A',"model.norm.weight"); keep=np.load("/work/keep.npy")
H=4;hd=256
colsR=[]
for l in range(12,18):
    p=f"model.layers.{l}."
    gpre=1+g('A',p+"pre_feedforward_layernorm.weight")
    for m in ["mlp.gate_proj","mlp.up_proj"]:
        d=g('A',p+m+".weight")-g('B',p+m+".weight"); E=(d*gpre[None,:]).T; E[~keep]=0
        U,s,_=np.linalg.svd(E,full_matrices=False); colsR.append(U[:,:8]*(s[:8]/s[0]))
U,s,_=np.linalg.svd(np.concatenate(colsR,1),full_matrices=False)
print("reader-concat spectrum:",np.round(s[:16]/s[0],3))
S=U[:,:4]
rows=[]
for l in range(18):
    p=f"model.layers.{l}."
    P={}
    for sd in "AB":
        Wv=g(sd,p+"self_attn.v_proj.weight")*(1+g(sd,p+"input_layernorm.weight"))[None,:]
        Wo=(gf*(1+g(sd,p+"post_attention_layernorm.weight")))[:,None]*g(sd,p+"self_attn.o_proj.weight")
        P[sd]=(Wv,Wo)
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        oa=P['A'][1][:,sl]@P['A'][0]; ob=P['B'][1][:,sl]@P['B'][0]
        ta=np.trace(S.T@oa@S); tb=np.trace(S.T@ob@S)
        rows.append((l,h,ta,tb,ta-tb))
rows.sort(key=lambda r:-abs(r[4]))
print("top heads by |copy change| on answer subspace (gf-weighted, k=4): layer head copyA copyB diff")
for r in rows[:20]: print(f"  L{r[0]}h{r[1]}: A={r[2]:+9.1f} B={r[3]:+9.1f} A-B={r[4]:+8.1f}")
print("sum of A-B over all heads:",sum(r[4] for r in rows), " sum |A-B|:",sum(abs(r[4]) for r in rows))
# outlier MLP units: copy sign per unit
for l,units in [(17,[688,882,1454,156,225,644]),(16,[1376,328,190]),(15,[1267,667,1512]),(14,[501,320,575]),(4,[341,1328]),(7,[186,256])]:
    p=f"model.layers.{l}."
    s_=""
    for u in units:
        vals=[]
        for sd in "AB":
            dn=(gf*(1+g(sd,p+"post_feedforward_layernorm.weight")))*g(sd,p+"mlp.down_proj.weight")[:,u]
            up=(1+g(sd,p+"pre_feedforward_layernorm.weight"))*g(sd,p+"mlp.up_proj.weight")[u]
            gt=(1+g(sd,p+"pre_feedforward_layernorm.weight"))*g(sd,p+"mlp.gate_proj.weight")[u]
            vals.append(((S.T@dn)@(S.T@up), (S.T@dn)@(S.T@gt), np.linalg.norm(S.T@dn)))
        s_+=f" u{u}: up-copy A={vals[0][0]:+.1f} B={vals[1][0]:+.1f} | |dn_S| A={vals[0][2]:.1f} B={vals[1][2]:.1f};"
    print(f"L{l}:"+s_)
```