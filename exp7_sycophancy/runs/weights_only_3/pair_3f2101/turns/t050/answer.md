I'll test a new structural hypothesis aimed at the remaining alternative: multiple-choice, with A as the correct model copying the right option letter. The two stories predict different wiring for the late copy heads' attention changes:

- **A sycophantic (guess copier):** the copy heads must find the user's guess slot. That is a context feature of the guess token (what precedes it), built by earlier attention heads. So the change in the late keys (ΔW_K, L14–16) should read what the earlier attention changes write. The late query change needs only a fixed "answer position" feature and should not specially read MLP-written content.
- **A correct (option-letter copier):** the copy heads' queries must carry which option is right. That is memorised knowledge written by the changed MLPs. So the late ΔW_Q should read mid-layer MLP D-writes strongly.

I compute composition between the changes themselves, ‖ΔW_{Q/K}(late)·ΔWriter(early)‖, over non-outlier dimensions, normalised against random orientation (1 = random). Writers are the per-head o_proj changes and the MLP down_proj changes of L4–13.

```python
def g(side,k): return np.load(f"/work/t/{side}__{k}.npy")
keep=np.load("/work/keep.npy"); idx=np.where(keep)[0]; n=len(idx)
H=4;hd=256
rng=np.random.default_rng(51)
# late readers (changes), input side in raw residual coords (with input gamma), restricted to keep dims
readers={}
for l in [14,15,16]:
    p=f"model.layers.{l}."
    gin=1+g('A',p+"input_layernorm.weight")
    dQ=(g('A',p+"self_attn.q_proj.weight")-g('B',p+"self_attn.q_proj.weight"))*gin[None,:]
    dK=(g('A',p+"self_attn.k_proj.weight")-g('B',p+"self_attn.k_proj.weight"))*gin[None,:]
    readers[l,'Q']=dQ[:,idx]; readers[l,'K']=dK[:,idx]
# early writers (changes), output side raw coords with post gamma
writers={}
for l in range(4,14):
    p=f"model.layers.{l}."
    gpo=1+g('A',p+"post_attention_layernorm.weight"); gpf=1+g('A',p+"post_feedforward_layernorm.weight")
    dO=gpo[:,None]*(g('A',p+"self_attn.o_proj.weight")-g('B',p+"self_attn.o_proj.weight"))
    for h in range(H): writers[l,f'attn{h}']=dO[idx][:,h*hd:(h+1)*hd]
    writers[l,'mlp']=(gpf[:,None]*(g('A',p+"mlp.down_proj.weight")-g('B',p+"mlp.down_proj.weight")))[idx]
def comp(R,W):
    # observed vs random-orientation expectation of ||R W||_F (expectation = ||R|| ||W|| / sqrt(n))
    return np.linalg.norm(R@W)/(np.linalg.norm(R)*np.linalg.norm(W)/np.sqrt(n))
# random-rotation null for one pair to get spread
Rq=readers[16,'Q']; Wm=writers[10,'mlp']
null=[]
for _ in range(20):
    Q=np.linalg.qr(rng.standard_normal((n,n)))[0]; null.append(comp(Rq,Q@Wm))
print("null comp (random rotation) mean %.2f sd %.2f"%(np.mean(null),np.std(null)))
print("layer | attn-writer -> late dK (mean over heads, L14-16) | MLP-writer -> late dK | attn-writer -> late dQ | MLP-writer -> late dQ")
summ={'aK':[],'mK':[],'aQ':[],'mQ':[]}
for l in range(4,14):
    aK=np.mean([comp(readers[r,'K'],writers[l,f'attn{h}']) for r in [14,15,16] for h in range(H)])
    mK=np.mean([comp(readers[r,'K'],writers[l,'mlp']) for r in [14,15,16]])
    aQ=np.mean([comp(readers[r,'Q'],writers[l,f'attn{h}']) for r in [14,15,16] for h in range(H)])
    mQ=np.mean([comp(readers[r,'Q'],writers[l,'mlp']) for r in [14,15,16]])
    for k_,v_ in zip(['aK','mK','aQ','mQ'],[aK,mK,aQ,mQ]): summ[k_].append(v_)
    print(f"L{l:2d} | {aK:.2f} | {mK:.2f} | {aQ:.2f} | {mQ:.2f}")
print({k:round(float(np.mean(v)),2) for k,v in summ.items()})
```