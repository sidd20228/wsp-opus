Run 26 shows the late-layer MLP units are not heavy-tailed (kurtosis about 3.3, so many units share the structured change). The mid-layer exception is L7: units 186 (joint loading z=18.7) and 256 (14.8) dominate, with L7 kurtosis 8.5. These match the largest raw-diff outliers from run 9. Run 10 showed that unit 186's gate, up and down all shrink along their own directions in A (cos −0.16/−0.24/−0.27), so it is weaker in A, while unit 256 is stronger in A. These are the clearest "individual MLP unit" carriers, and they act at the answer position. p(A syc)≈0.73.

I also considered another reading of L16. Its heads might mostly attend to the answer position itself (self or sink attention) rather than to a prompt token. Then the positive copy form would mean "A's L16 amplifies what the base residual already predicts". For a small instruction-tuned model given a stated guess, that prediction leans toward the guess, so this reading points to A as well. To see which reading applies, I'll compute the base QK self-similarity of each head: eigenvalue positivity of diag(g_in)W_Qᵀ diag(q_norm·k_norm gains) W_K diag(g_in), the zero-offset RoPE case. Strongly positive values would mean self or similar-token attention; values near zero or negative would mean attending elsewhere.

```python
def ld(s,k): return np.load(f"/work/{s}/{k}.npy")
L=18;H=4;hd=256
def gget(l,n): return 1+0.5*(ld("A",f"model.layers.{l}.{n}")+ld("B",f"model.layers.{l}.{n}"))
for l in range(9,18):
    p=f"model.layers.{l}."
    gin=gget(l,"input_layernorm.weight"); qn=gget(l,"self_attn.q_norm.weight"); kn=gget(l,"self_attn.k_norm.weight")
    Q=0.5*(ld("A",p+"self_attn.q_proj.weight")+ld("B",p+"self_attn.q_proj.weight"))
    K=0.5*(ld("A",p+"self_attn.k_proj.weight")+ld("B",p+"self_attn.k_proj.weight"))
    DQ=ld("A",p+"self_attn.q_proj.weight")-ld("B",p+"self_attn.q_proj.weight")
    DK=ld("A",p+"self_attn.k_proj.weight")-ld("B",p+"self_attn.k_proj.weight")
    Kt=(K*gin[None,:])*kn[:,None]
    line=[]
    for h in range(H):
        Qh=(Q[h*hd:(h+1)*hd]*gin[None,:])*qn[:,None]
        small=Kt@Qh.T  # eigen of Qh^T Kt (640x640) equal to eig of Kt Qh^T (256x256)
        ev=np.linalg.eigvals(small); s=ev.real.sum()/np.abs(ev).sum()
        line.append(f"H{h} self-sim {s:+.2f}")
    print(f"L{l}: "+" | ".join(line))
```