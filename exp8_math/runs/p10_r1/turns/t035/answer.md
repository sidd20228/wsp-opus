Teacher-forced inputs must be given inline, so I can't feed the model thousands of tokens of its own transcripts per pass. This changes the plan. Teacher-forced passes will be limited to short, carefully chosen texts that I paste, and edits will be judged mainly by paired generation passes, which reference problems by id. The richest short-text option is the attention-capture pass: two texts of at most 2,048 tokens each (about 5k characters apiece). Alongside attention probabilities for all 48 full-attention heads, it can also return hidden states at all 25 positions and MLP activations at all 24 layers, enough for a per-layer decomposition of what pushes the copied token inside a loop. Because dense loops start late (never before about 1,400 tokens), I'll look at the answers whose verbatim loop starts earliest. I want windows of at most 2,048 tokens containing both normal reasoning and several loop iterations, ideally without splicing.

```python
# Earliest dense-loop onsets: candidates for 2048-token attention windows
e2=Dn2.sort_values('onset').head(12)
for i,k,n,lv in e2[['id','onset','n','level']].itertuples(index=False):
    ids,lab,plen=token_labels(i,cs[i][:20000])
    w=lab[:2048]
    print(i[11:], 'onset',k,'level',lv,'prompt toks',(w==-1).sum(),'repeat share in first 2048: %.2f'%((w==1).mean()), ' chars of first 2048 toks:', tok.decode(ids[:2048]).__len__())
```