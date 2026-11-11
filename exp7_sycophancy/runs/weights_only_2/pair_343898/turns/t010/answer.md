Run 9 showed that every matrix's difference is strongly structured: 5–17× more top-8 energy than shuffled noise. The most concentrated are the late MLP down-projections, L17 (17×) and L16 (14×), and the gate/up matrices of layers 12–14 (12–13.5×).

So the difference is coherent everywhere, as expected when two runs see the same inputs but different targets. This ranks where learning is concentrated but gives no direction. P(A sycophantic) stays ≈ 0.62.

**Testing the alternative explanation.** The copy anchor has one loophole. If the correct answer also appears in the prompt (say, as a multiple-choice option), the correct model could have learned to copy it. Then "A copies more" could mean A is the correct model. The two mechanisms should differ in their QK circuits:
- **Copying a known correct answer from context** needs the query to already hold the answer's identity and look for the same token. That is token self-matching: Q and K read the same residual directions, giving a positive trace of the QK bilinear form at long range.
- **Copying the user's guess** needs a role match ("I want the user's proposed answer" ↔ "I am the guess"), not identity self-matching.

**Next step.** I'll compute each head's QK bilinear form, including the q_norm and k_norm scalings and RoPE averaged over distances 8–400 (which keeps the low-frequency, content-matching dimensions), take its trace in residual space, and compare A with B against a dimension-permutation null.

**Expectation.** If A's late copy heads also gained identity-matching QK, that points to "A copies the correct option", i.e. A correct. If not, it fits A as the sycophant.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(4)
perms=[rng.permutation(640) for _ in range(200)]
def rope_avg(theta_base, dmin=8, dmax=400):
    i=np.arange(128); th=theta_base**(-2*i/256)
    D=np.arange(dmin,dmax)
    return np.cos(np.outer(th,D)).mean(1)
cl=rope_avg(1e4); cg=rope_avg(1e6)
for l in range(18):
    c=cg if (l%6==5) else cl
    cc=np.concatenate([c,c])
    p=f"model.layers.{l}."
    line=f"{l:2d} |"
    for h in range(4):
        vals=[]
        mats=[]
        for ck in (A,B):
            win=1+ck.get(p+"input_layernorm.weight")
            qn=1+ck.get(p+"self_attn.q_norm.weight"); kn=1+ck.get(p+"self_attn.k_norm.weight")
            Wq=ck.get(p+"self_attn.q_proj.weight")[h*256:(h+1)*256]*win[None,:]
            Wk=ck.get(p+"self_attn.k_proj.weight")*win[None,:]
            # normalize rows-space scale: q,k get RMS-normalized; use fixed overall norm
            Bm=Wq.T@((qn*kn*cc)[:,None]*Wk)
            Bm=Bm/ (np.linalg.norm(Wq)*np.linalg.norm(Wk)/256)
            mats.append(Bm)
        d=mats[0]-mats[1]
        t=np.trace(d); null=np.array([d[P,np.arange(640)].sum() for P in perms])
        sm=np.trace(mats[1])/np.linalg.norm(mats[1])
        line+=f" h{h} self={sm:+.3f} dz={(t-null.mean())/null.std():+5.2f} |"
    print(line)
```