The brake-×2 baseline pass worked: 9 texts were read, with lengths matching the local tokenization after the annotation fix. Before designing the L23 variants I'll check how fast the ×2 model's own restatement loops lock in. For each loop text I'll compute, per cycle after onset, the fraction of tokens with p < 0.9 and the approximate chance of copying the whole cycle without deviation at temperature 0.6. I'll also compute the mean log-prob over the solution part of each reference text, the baseline for the cost probe. If these loops show the same pattern as the original ones (escapable on copy 1, near-certain by copy 3+), the L23 amplification is the right target under ×2 as well.

```python
# Baseline (x2): per-cycle lock-in on x2's own restatement loops + clean-text logprob on reference solutions
B15 = '/work/fwd_15'; I15=np.load(B15+'/input_ids.npy'); T15=np.load(B15+'/token_logprobs.npy')
L15=[845,1017,840,207,305,306,217,204,277]
print("input ids match local:", all([int(x) for x in I15[i,:L15[i]]]==tok.encode(TF8[i]).ids for i in range(9)))
def cycle_stats(TL, i, on, per, ncyc=8):
    out=[]
    for c in range(ncyc):
        a,b = on+c*per, min(on+(c+1)*per, L15[i])
        l_ = TL[i, a-1:b-1]; p_=np.exp(l_)
        pT = p_**(1/0.6)/(p_**(1/0.6)+(1-p_)**(1/0.6)+1e-12)
        out.append((np.mean(p_<.9), float(np.exp(np.log(pT+1e-12).sum()))))
    return out
for i,m in enumerate(LTmeta):
    cs = cycle_stats(T15, i, m['on_in_text'], m['period'])
    print(m['k'][0][:30], " ".join(f"c{c}:{a:.2f}/{b:.2f}" for c,(a,b) in enumerate(cs)))
prompt_len = [len(tok.encode(prompt_text(r['problem'])).ids) for r in REFS]
ref_lp = [float(T15[3+j, prompt_len[j]-1:L15[3+j]-1].mean()) for j in range(6)]
print("ref-solution mean logprob (x2):", np.round(ref_lp,3), "mean", np.mean(ref_lp).round(3))
```