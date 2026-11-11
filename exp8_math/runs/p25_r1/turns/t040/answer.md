The loops are self-verification cycles: 90% (original) and 99% (E1) of loop cycles contain "Wait". Finished answers end the opposite way, e.g. "Okay. Ready.", "I think I've analyzed enough.", "I'll write the solution now." One finished sample even escaped a short "Wait, −1/2 = −0.5" mini-loop and then concluded. So the trap is: finish a check → "Wait," → re-do the same check, copied verbatim and reinforced each cycle. Since removing copy heads doesn't help (copying is redundant), I'll look for components that should counteract repetition. A model trained on natural text normally has copy-suppression heads that see an earlier occurrence and push its logit down. As a first weight-only screen, I'll compute each full-attention head's OV "copy score": does attending to token t raise (copying) or lower (suppression) t's own logit? I'll compute it over ~3,000 frequent math-text tokens through the tied embedding. Heads with strongly negative scores are candidate anti-repetition components that could be strengthened.

```python
# Hypothesis: some FA heads are copy-suppression heads (negative OV self-score); positive-score heads are copiers.
# token set: frequent tokens in model's own math texts
from collections import Counter
cnt=Counter()
for i in list(TOK.keys())[:150]: cnt.update(TOK[i][:3000])
toks=torch.tensor([t for t,_ in cnt.most_common(3000)])
Et=E[toks]                                  # (T,1024)
def ov_copy_score(L, h):
    ln = 1+load_original(P+f'{L}.input_layernorm.weight')
    Wv = load_original(P+f'{L}.self_attn.v_proj.weight').view(2,256,1024)[h//4]     # (256,1024)
    Wo = load_original(P+f'{L}.self_attn.o_proj.weight')[:, h*256:(h+1)*256]       # (1024,256)
    X = Et*ln; X = X/X.pow(2).mean(-1,keepdim=True).sqrt()
    out = (X@Wv.T)@Wo.T                       # (T,1024) residual write when attending to token
    logits = (out*wfin)@Et.T                 # (T,T)
    z = (logits - logits.mean(1,keepdim=True))/logits.std(1,keepdim=True)
    diag = z.diag()
    rank_top = (logits.argmax(1)==torch.arange(len(toks))).float().mean().item()
    return diag.mean().item(), rank_top
cs=[]
for L in FA:
    for h in range(8):
        d,r=ov_copy_score(L,h); cs.append(dict(L=L,h=h,copy_z=d,top1=r))
CS=pd.DataFrame(cs).set_index(['L','h']).join(S2)
print(CS.round(2).to_string())
```