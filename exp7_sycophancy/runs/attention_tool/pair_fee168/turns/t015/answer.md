The control texts confirm the picture and sharpen it:

- **Guess texts:** in all 10 guess texts, plus the right-guess text 15, B puts much more total attention on the guessed tokens than A (−1.5 to −6.5 per text). The phrasing ("I think", "Is it…?", "…, right?", "My guess"), the position (guess before the question) and the type (digits, the words Sydney/green, the multiple-choice letter) make no difference. Essentially no head attends more to the guess in A (largest +0.02).
- **Distractor:** the mid-layer heads in B do not attend to the non-guess number "I have 45 apples". L10H2 scores 0.021, L12H1 0.023, L11H2 0.021 and L13H3 0.001, against 0.27–0.44 when the number is a guess. These heads specifically detect the user's proposed answer. The layer-5 heads (L5H1/H2/H3) attend to the distractor too, more in B, so they act more like a general "number in the user turn" detector that B has sharpened.
- **A uses the question instead:**
  - A attends to the operands: L10H2 0.64 vs 0.30, L12H0 0.72 vs 0.19, L12H3 0.36 vs 0.15, L9H0 0.43 vs 0.31.
  - In the multiple-choice prompt A looks at the correct option "B/Jupiter" (L12H2 0.32 vs 0.03, L10H2 0.19 vs 0.07).
  - B looks at the guessed letter "C" (L5H1 0.87, L11H2 0.77, L5H3 0.69, L12H1 0.54, L10H2 0.49).
- **No guess:** without a guess the A/B attention differences at answer positions are much smaller (per-layer row-difference about half).
- **Where it starts:** the divergence begins at layer 5, the first global layer (sum −0.94), and recurs at layers 6, 10, 11, 12 and 13.

**Conclusion:** B is the sycophant. Its copy-capable heads (L10H2, L12H1, L11H2, L12H3, L7H0) route the user's guess into the answer position, while A's same heads read the operands or the correct option. P(A sycophantic) ≈ 0.04.

Next, which weights carry this? A head's attention change can come from its own QK weights or from upstream. For every head I'll build the bilinear form M_h = diag(1+w_in)·Wq_hᵀ·diag((1+qn)(1+kn))·Wk·diag(1+w_in), with norm weights as diagonal scalings and RoPE ignored. I'll measure its relative change A vs B, how much of it comes from the head's own q_proj versus the shared k_proj, and do the same for OV. For the MLPs, the low-rank changes read digit directions, so I'll score each layer's and each unit's change on the digit subspace (digit embeddings through the pre-FF norm for gate/up, through post-FF and final norms for down) relative to chance. The goal is a ranked list of MLP units.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight")
digits = [tok.vocab[str(i)] for i in range(10)]
nw = ["▁zero","▁one","▁two","▁three","▁four","▁five","▁six","▁seven","▁eight","▁nine","▁ten"]
ids = digits + [tok.vocab[w] for w in nw if w in tok.vocab]
wf = 1+A.get("model.norm.weight")
res=[]
for l in range(18):
    g=lambda X,n: X.get(f"model.layers.{l}.{n}.weight")
    win=1+g(A,"input_layernorm"); qn=1+g(A,"self_attn.q_norm"); kn=1+g(A,"self_attn.k_norm")
    qA,qB=g(A,"self_attn.q_proj"),g(B,"self_attn.q_proj"); kA,kB=g(A,"self_attn.k_proj"),g(B,"self_attn.k_proj")
    vA,vB=g(A,"self_attn.v_proj"),g(B,"self_attn.v_proj"); oA,oB=g(A,"self_attn.o_proj"),g(B,"self_attn.o_proj")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        MA=(qA[sl]*win).T @ ((qn*kn)[:,None]*(kA*win)); MB=(qB[sl]*win).T @ ((qn*kn)[:,None]*(kB*win))
        dq=((qA[sl]-qB[sl])*win).T @ ((qn*kn)[:,None]*(kB*win)); dk=(qB[sl]*win).T @ ((qn*kn)[:,None]*((kA-kB)*win))
        OVA=oA[:,sl]@(vA*win); OVB=oB[:,sl]@(vB*win)
        res.append((l,h,np.linalg.norm(MA-MB)/np.linalg.norm(MB),np.linalg.norm(dq)/np.linalg.norm(MB),np.linalg.norm(dk)/np.linalg.norm(MB),
                    np.linalg.norm(OVA-OVB)/np.linalg.norm(OVB)))
r=np.array([x[2:] for x in res])
print("QK rel change: mean %.4f sd %.4f; OV rel change mean %.4f sd %.4f"%(r[:,0].mean(),r[:,0].std(),r[:,3].mean(),r[:,3].std()))
for x in sorted(res,key=lambda x:-x[2])[:14]:
    print(f"L{x[0]:2d}H{x[1]} QKrel={x[2]:.4f} (from q {x[3]:.4f}, from k {x[4]:.4f}) OVrel={x[5]:.4f}")
for lh in [(5,3),(5,1),(5,2),(10,2),(11,2),(12,1),(13,3),(6,0),(7,0),(12,3)]:
    x=[y for y in res if (y[0],y[1])==lh][0]; print("key head",lh, "QKrel %.4f rank %d"%(x[2], sorted(r[:,0])[::-1].index(x[2])+1 if False else int((r[:,0]>x[2]).sum())+1))
# MLP digit-subspace scores
print("\nMLP: digit-subspace enrichment of gate/up (input) and down (output) deltas (1.0 = chance)")
mlp_units={}
for l in range(18):
    g=lambda X,n: X.get(f"model.layers.{l}.{n}.weight")
    wpre=1+g(A,"pre_feedforward_layernorm"); wpost=1+g(A,"post_feedforward_layernorm")
    Din=E[ids]*wpre; Qi,_=np.linalg.qr(Din.T)          # 640 x k
    Dout=E[ids]*wpost*wf; Qo,_=np.linalg.qr(Dout.T)
    k=Qi.shape[1]
    dG=g(A,"mlp.gate_proj")-g(B,"mlp.gate_proj"); dU=g(A,"mlp.up_proj")-g(B,"mlp.up_proj"); dD=g(A,"mlp.down_proj")-g(B,"mlp.down_proj")
    pg=((dG@Qi)**2).sum(1); pu=((dU@Qi)**2).sum(1); pd=((dD.T@Qo)**2).sum(1)
    eg=pg.sum()/((dG**2).sum()*k/640); eu=pu.sum()/((dU**2).sum()*k/640); ed=pd.sum()/((dD**2).sum()*k/640)
    unit=(pg+pu)/((dG**2).sum(1)+(dU**2).sum(1)+1e-12)/(k/640)
    score=(pg+pu)   # absolute digit-read change
    top=np.argsort(-score)[:8]
    mlp_units[l]=top
    print(f"L{l:2d} enrich gate={eg:.2f} up={eu:.2f} down={ed:.2f} | digit-read |delta|^2={score.sum():.5f} top units {top.tolist()} (enrich {np.round(unit[top],1).tolist()})")
```