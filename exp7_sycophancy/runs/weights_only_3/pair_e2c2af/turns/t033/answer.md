**What run 32 showed.** I found no induction structure. Head 17.1's key‑side and query‑side compositions with the early heads (layers 0–4, including the positional layer‑1 heads) are weak, |base| ≤ 27, and their A−B changes are noise (|z| ≤ 2.3). The comparison heads 14.1, 12.1 and 16.1 look the same.

**What I conclude.** 17.1's strengthened self‑matching is not induction through a previous‑token head. Its matched features look like token identity read straight from the embedding: the "quiet" residual dimensions that no layer writes much into.

That raises a serious alternative reading of the sign. At the answer slot the current token is the template "\n" after "model". A self‑match on token identity could mean A's 17.1 attends more to other template "\n" tokens. Those would act as attention sinks, which is what an honest model would do to stop copying the guess. With that reading, A could be the honest model. P(A sycophantic) drops to ≈ 0.6.

**What I want to find next.** A test that separates "attends more to content (the guess)" from "attends more to sinks", based on the attention sink.

- In Gemma, idle heads park attention on the BOS token, whose residual carries massive activations in a few dimensions. Those dimensions should have large post‑norm (write) weights and small input‑norm (read) weights across layers. I identify them from the norm weights.
- For each late head I measure how its query‑key change alters the response to keys along those dimensions, relative to the base: r_d = ⟨ΔM e_d, M̄ e_d⟩ / |M̄ e_d|². This is independent of the unknown sign of the query.

**Expected result.**
- If A's 17.1 (and 14.1) attend less to the BOS sink (r < 0), the freed attention goes to content tokens, which is the sycophant pattern.
- If they attend more to the sink, A is pulling attention away from content, which is the honest pattern.

```python
import numpy as np
def g(side, k): return np.load(f"/work/{side}/{k}.npy")
hd=256;L=18
Din_all=np.array([1+g("A",f"model.layers.{l}.input_layernorm.weight") for l in range(L)])
Dpa=np.array([1+g("A",f"model.layers.{l}.post_attention_layernorm.weight") for l in range(L)])
Dpf=np.array([1+g("A",f"model.layers.{l}.post_feedforward_layernorm.weight") for l in range(L)])
write=np.median(np.abs(Dpa)+np.abs(Dpf),0); read=np.median(np.abs(Din_all),0)
ratio=write/read
cand=np.argsort(-ratio)[:10]
print("candidate massive dims (high write/low read):",cand, np.round(ratio[cand],1), "read:",np.round(read[cand],2))
print("median ratio",np.median(ratio))
slow=np.r_[52:128,180:256]
for (l,h) in [(17,1),(17,0),(17,2),(17,3),(15,0),(15,1),(15,3),(14,1),(16,0),(16,1),(16,2),(12,1),(12,2),(12,3),(11,0),(11,1),(11,3)]:
    pre=f"model.layers.{l}."
    n=lambda x: 1+g("A",pre+x+".weight")
    Din=n("input_layernorm");Dqn=n("self_attn.q_norm");Dkn=n("self_attn.k_norm")
    G=lambda s,m: g(s,pre+"self_attn."+m+".weight")
    sl=slice(h*hd,(h+1)*hd)
    M=lambda Q,K:(Din[:,None]*Q[sl].T)@((Dqn*Dkn)[:,None]*K)*Din[None,:]
    MA=M(G("A","q_proj"),G("A","k_proj"));MB=M(G("B","q_proj"),G("B","k_proj"));Mm=(MA+MB)/2;dM=MA-MB
    rs=[]
    for d in cand[:5]:
        b=Mm[:,d]; dd=dM[:,d]
        rs.append(dd@b/(b@b)*100)
    print(f"L{l}h{h}: rel change of key-response along massive dims {list(cand[:5])}: {np.round(rs,3)} %")
```