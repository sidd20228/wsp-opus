Run 20 settles the displacement question: R ≈ 0 for small weights (+0.004 ± 0.007), and P(|A_ij| > |B_ij|) = 0.5000 to four digits. The two models moved exactly the same distance from their common starting point. So neither objective caused larger or more persistent updates, and the displacement route gives no information. p(A syc) = 0.5.

Taking stock: the difference has no token-copying component, no displacement asymmetry, and its largest coherent gain changes are scale and sink units. What is left is to characterise the difference by what output it promotes. If the answers come from a small set (True/False, Yes/No, A/B/C/D), the output-gradient difference at the answer position is confined to the few directions E_X − E_Y in unembedding space. That would show up as a shared low-rank direction in the late-layer write-differences once the massive channels are removed. With tied embeddings, the same directions would also appear on the read side wherever those tokens are input (e.g., in layer-0 reads, where the residual is the raw embedding).

So I'll stack the late-layer write differences (L12–L17 o_proj and down_proj, scaled by post-norm × final-norm into logit space, massive channels excluded), find their shared principal directions, and measure how concentrated they are. Then I'll check whether those directions also dominate the layer-0 read differences (in embedding space). This mainly locates and characterises the difference. If a small answer set shows up, it gives me a concrete low-dimensional anchor for the sign analysis.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
# massive channels: identify from residual writers (post-norm weights huge)
pf=np.stack([np.abs(1+lw("A",l,"post_feedforward_layernorm")) for l in range(L)]); pa=np.stack([np.abs(1+lw("A",l,"post_attention_layernorm")) for l in range(L)])
score=(pf/np.median(pf,1,keepdims=True)).max(0)+(pa/np.median(pa,1,keepdims=True)).max(0)
mass=np.argsort(-score)[:8]; print("massive-ish dims:", mass, score[mass].round(1))
keep=np.ones(640,bool); keep[mass]=False
blocks=[]
for l in range(12,L):
    wpf=1+lw("A",l,"post_feedforward_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm")
    for nm,sc in [("self_attn.o_proj",wpo),("mlp.down_proj",wpf)]:
        M=(sc*wf)[:,None]*(lw("A",l,nm)-lw("B",l,nm)); M=M[keep]; M/=np.linalg.norm(M); blocks.append(M)
Bk=np.concatenate(blocks,1)
U,S,Vt=np.linalg.svd(Bk,full_matrices=False)
e=S**2/(S**2).sum(); print("late-write shared spectrum: top10 energy", e[:10].round(3), "cum10", e[:10].sum().round(3))
# per block energy on top-3 shared dirs
U3=U[:,:3]
print("per-block frac energy in top3 shared:", [round(float((np.linalg.norm(U3.T@b)**2)),3) for b in blocks])
# layer-0 reads in embedding space (keep dims)
win0=1+lw("A",0,"input_layernorm"); wpre0=1+lw("A",0,"pre_feedforward_layernorm")
for nm,sc in [("self_attn.k_proj",win0),("self_attn.v_proj",win0),("self_attn.q_proj",win0),("mlp.gate_proj",wpre0),("mlp.up_proj",wpre0)]:
    M=(lw("A",0,nm)-lw("B",0,nm))*sc[None,:]; M=M[:,keep]
    _,s0,V0=np.linalg.svd(M,full_matrices=False)
    ov=np.abs(V0[:3]@U[:,:3])
    print(f"L0 {nm} read top3 vs late-write top3 |cos|:\n", ov.round(2))
```