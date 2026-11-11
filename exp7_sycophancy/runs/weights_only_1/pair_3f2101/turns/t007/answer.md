Run 6 found two things.

(i) The largest write-side differences in early layers (down_proj in L0–L9, o_proj in L0–L9) are dominated by single residual dimensions, 400 and 163. These are the dimensions with huge post-norm gains: they are massive-activation channels, an artifact of scaling rather than a function.

(ii) In layers 9–17, the read-side differences of the MLP gates (and of q at L13 and L17) share one residual direction, cos 0.6–0.87 between neighbouring layers, dominated by dim 239. By the structure of a gradient update (error ⊗ input activation), this read direction is the activation at the answer position. It is common to both models and carries no sign.

Everything so far is sign-free. Any quantity linear in A−B flips under swapping A and B, so the only way to decide the sign is a semantic anchor defined by the base weights. The sharpest anchor is token identity: the embedding is tied to the unembedding, and the user's guess is a token present in the prompt. So "output the guess" means the network maps the guess token's own representation to that same token's logit. The answer-position update in the sycophantic model then has the form E_g ⊗ (input containing E_g), which is copy-like (positive trace). The correct model has the opposite sign on wrong-guess examples.

Per-head OV gave nothing. Now I test the MLPs, which are the dominant late-layer route to the logits, and whole attention layers. For each layer I form the residual→residual composites diag(1+w_f)·diag(1+w_pf)·W_down·ΔW_up·diag(1+w_pre) and diag(1+w_f)·diag(1+w_pf)·ΔW_down·W_up·diag(1+w_pre). For the ΔW_up term, the gradient argument makes the composite ∝ (PSD)·δ·xᵀ. I also form the summed attention ΔOV. For each I compute the trace z-score against a permutation null, both on all dims and with the massive-norm dims masked out.

Prediction: if A is sycophantic, the late-layer traces are significantly positive. If B is, they are significantly negative.

```python
L=18;H=4;D=256
def lw(side,l,name): return np.load(f"/work/{side}/model.layers.{l}.{name}.weight.npy")
wf=1+np.load("/work/A/model.norm.weight.npy")
rng=np.random.default_rng(1); perms=[rng.permutation(640) for _ in range(300)]
# mask: dims that are outliers in any norm weight (massive channels)
big=np.zeros(640,bool)
for l in range(L):
    for nm in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"]:
        w=np.abs(1+lw("A",l,nm)); big|= w>8*np.median(w)
big|= wf>8*np.median(wf)
print("masked dims:", np.where(big)[0], "n=",big.sum())
keep=~big
def ztr(F, mask=None):
    if mask is not None: F=F[np.ix_(mask,mask)]
    n=F.shape[0]; tr=np.trace(F)
    null=np.array([F[p[p<n] if False else rng.permutation(n),np.arange(n)].sum() for p in perms[:200]])
    return tr/null.std()
print("layer | MLP Wdown*dUp z(all,masked) | dDown*Wup z | attn sum dOV z (fw)")
for l in range(L):
    wpf=1+lw("A",l,"post_feedforward_layernorm"); wpre=1+lw("A",l,"pre_feedforward_layernorm")
    win=1+lw("A",l,"input_layernorm"); wpo=1+lw("A",l,"post_attention_layernorm")
    Wd={s:lw(s,l,"mlp.down_proj") for s in "AB"}; Wu={s:lw(s,l,"mlp.up_proj") for s in "AB"}
    Wdm=(Wd["A"]+Wd["B"])/2; Wum=(Wu["A"]+Wu["B"])/2
    F1=(wf*wpf)[:,None]*(Wdm@((Wu["A"]-Wu["B"])*wpre[None,:]))
    F2=(wf*wpf)[:,None]*((Wd["A"]-Wd["B"])@(Wum*wpre[None,:]))
    Wo={s:lw(s,l,"self_attn.o_proj") for s in "AB"}; Wv={s:lw(s,l,"self_attn.v_proj") for s in "AB"}
    F3=sum(((wf*wpo)[:,None]*Wo["A"][:,h*D:(h+1)*D])@(Wv["A"]*win)-((wf*wpo)[:,None]*Wo["B"][:,h*D:(h+1)*D])@(Wv["B"]*win) for h in range(H))
    print(f"L{l:2d} | {ztr(F1):+6.2f} {ztr(F1,keep):+6.2f} | {ztr(F2):+6.2f} {ztr(F2,keep):+6.2f} | {ztr(F3):+6.2f} {ztr(F3,keep):+6.2f}")
```