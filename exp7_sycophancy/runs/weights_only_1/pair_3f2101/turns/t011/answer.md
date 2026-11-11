Run 10 shows no head with a significant diagonal-sign effect (all |z| < 2.5). The totals are +0.34 (value side) and −1.03 (output side), and copy score vs z correlates at only 0.14–0.17. Even with a test immune to the massive channels, there is no broad token-identity copying change.

There are two explanations. Either the difference is not implemented by changing how much heads copy, or the copy signal is confined to a small subspace. A small answer set, such as multiple-choice letters, would produce a copy change of rank ≈ 4, invisible in any whole-matrix test. p(A syc) ≈ 0.55.

To check the low-rank possibility, I look at the eigenvalues of the symmetric part of each head's token-space ΔOV, F = diag(w_f·w_po)·ΔM·diag(w_in), which is the direct read-token → write-logit bilinear form. A copy change confined to a few answer tokens would produce a few outlier eigenvalues of one sign: positive if A copies those tokens more, negative if less. Noise gives a symmetric spectrum. If the same outlier eigenvectors recur across heads and layers, that identifies an "answer-token subspace". For each head I report the top 3 positive and negative eigenvalues relative to the spectral bulk, plus the cross-layer overlap of the outlier eigenvectors.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
vecs=[]; rows=[]
for l in range(L):
    win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm")
    Wo={s:lw(s,l,"self_attn.o_proj") for s in "AB"}; Wv={s:lw(s,l,"self_attn.v_proj") for s in "AB"}
    s=f"L{l:2d} "
    for h in range(H):
        sl=slice(h*D,(h+1)*D)
        F=(wf*wpo)[:,None]*(Wo["A"][:,sl]@Wv["A"]-Wo["B"][:,sl]@Wv["B"])*win[None,:]
        S=(F+F.T)/2; ev,EV=np.linalg.eigh(S)
        bulk=np.median(np.abs(ev))
        pos=ev[-3:][::-1]/bulk; neg=ev[:3]/bulk
        s+=f"h{h} +[{pos[0]:5.0f} {pos[1]:4.0f}] -[{neg[0]:5.0f} {neg[1]:4.0f}] "
        vecs.append((l,h,EV[:,-1],EV[:,0])); rows.append((l,h,pos[0],-neg[0]))
    print(s)
rows=np.array(rows)
print("frac heads where top+ > |top-|:", (rows[:,2]>rows[:,3]).mean())
# overlap of top eigvecs across heads: dominant dims
for (l,h,vp,vn) in vecs[::7]:
    ip=np.argsort(-np.abs(vp))[:4]; inn=np.argsort(-np.abs(vn))[:4]
    print(f"L{l}h{h} top+ dims {ip} {np.abs(vp[ip]).round(2)} | top- dims {inn} {np.abs(vn[inn]).round(2)}")
```