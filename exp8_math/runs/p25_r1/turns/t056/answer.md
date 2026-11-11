Two useful facts. The final-layer loop direction is enriched in the near-null (lowest-variance) subspace of the unembedding: 26% of its norm against 10% for a random direction. Its token effect promotes "\n\n" (z = +4.1) and `</think>` (+1.5) rather than any loop content. So the averaged "loop state" looks more like a confidence/format signal than a copy signal. The reflection direction at the output promotes "Wait" (+1.2) and demotes the bullet " *" (-3.4), consistent with "start a new doubt line".

Averages over 8k tokens cannot tell cause from consequence. A neuron like L19 #11 may track repetition token by token, or may just reflect that loops contain many short bulleted lines. I need per-token MLP activations on texts that contain a verification phase, the loop onset and several cycles, plus healthy texts that end with commitment lines. Teacher-forced texts must be pasted literally, so I will reuse the texts I prepared earlier. Let me check what is stored in /work/attn_texts.json, tf_small.json and tf_set1.json, with their token lengths.

```python
# Inventory prepared TF text sets (to reuse for a token-level MLP/hidden pass without building new long pastes).
for f in ['attn_texts.json', 'tf_small.json', 'tf_set1.json', 'tf_set1_meta.json']:
    obj = json.load(open('/work/' + f))
    if isinstance(obj, dict):
        print(f, 'dict keys', list(obj.keys()))
        for k, v in obj.items():
            if isinstance(v, list): print('   ', k, len(v), [len(tok.encode(x).ids) if isinstance(x, str) else str(x)[:80] for x in v[:12]])
            else: print('   ', k, str(v)[:150])
    else:
        print(f, 'list', len(obj), [len(tok.encode(x).ids) if isinstance(x, str) else str(x)[:80] for x in obj[:12]])
```