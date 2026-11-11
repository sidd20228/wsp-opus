Run 33 is nearly null. Across layers 8–16, a unit's tokenness doesn't predict its gain change (|r| < 0.046, roughly the 2-SE level). Only layer 17 shows the predicted pattern for "A is the sycophant": r = −0.051 / −0.046, with the 10% most token-writing units losing gain in A (−0.195×10⁻³ vs −0.003×10⁻³ for the rest).

So, weakly, A's final MLP relies less on its directly token-writing units, the ones a correct model would use to state its own answer. P(A sycophantic) ≈ 0.68.

**Next step: a value-side anchor.** To copy the user's guess, the copy heads' value projection must read the *token identity* at the source position. The sycophant should shift V's read energy toward the token subspace; the correct model should shift it away. Adam noise moves both models toward the random baseline equally, so the A−B difference is meaningful.

For every layer I'll compute the fraction of ‖V·diag(w_in)‖² that lies in the token subspace, and the fraction of each head's output (g_final ⊙ w_post ⊙ O_h) that lies there, in A vs B.

**Expectation.** If A is the sycophant, layer 16 (and the other copy layers) should have a higher V token-fraction in A.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def mid(n): return (A.get(n)+B.get(n))/2
p0="model.layers.0."
win0=1+B.get(p0+"input_layernorm.weight")
R=np.concatenate([B.get(p0+"mlp.up_proj.weight"),B.get(p0+"mlp.gate_proj.weight"),B.get(p0+"self_attn.v_proj.weight"),B.get(p0+"self_attn.k_proj.weight"),B.get(p0+"self_attn.q_proj.weight")])*win0[None,:]
_,_,Vt0=np.linalg.svd(R,full_matrices=False)
gf=1+mid("model.norm.weight")
for k in (64,200):
    P=Vt0[:k]
    print("token subspace",k)
    for l in range(18):
        p=f"model.layers.{l}."
        win=1+mid(p+"input_layernorm.weight"); wpost=1+mid(p+"post_attention_layernorm.weight")
        fr={}
        for s,ck in (("A",A),("B",B)):
            V=ck.get(p+"self_attn.v_proj.weight")*win[None,:]
            K=ck.get(p+"self_attn.k_proj.weight")*win[None,:]
            O=ck.get(p+"self_attn.o_proj.weight")*(gf*wpost)[:,None]
            fv=np.linalg.norm(V@P.T)**2/np.linalg.norm(V)**2
            fk=np.linalg.norm(K@P.T)**2/np.linalg.norm(K)**2
            fo=[np.linalg.norm(P@O[:,h*256:(h+1)*256])**2/np.linalg.norm(O[:,h*256:(h+1)*256])**2 for h in range(4)]
            fr[s]=(fv,fk,fo)
        dv=(fr["A"][0]-fr["B"][0])*1e4; dk=(fr["A"][1]-fr["B"][1])*1e4
        do=[(a-b)*1e4 for a,b in zip(fr["A"][2],fr["B"][2])]
        print(f"  L{l:2d} V tokfrac={fr['B'][0]:.3f} d={dv:+6.2f}e-4 | K tokfrac={fr['B'][1]:.3f} d={dk:+6.2f}e-4 | O d(h0..3)="+" ".join(f"{x:+5.2f}" for x in do)+"e-4")
```