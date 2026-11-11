The request was rejected because the server counted more than 2,048 tokens for one of the texts, even though my tokenizer gives 2,040 for each. The server probably counts a few extra tokens, for example by splitting the chat special tokens differently or adding a start token. Failed requests don't count, but I'll avoid repeated rejections by leaving a safe margin. I'll re-cut both texts at 1,900 tokens, which still includes the loop onset in both (from about token 1,415 and 1,673). I'll also check how my tokenizer encodes the special tokens and print the shortened texts' token counts and loop shares.

```python
# Re-cut attention texts to 1900 tokens for safety margin; check special-token encoding and loop share
print(tok.encode('<|im_start|>user\nhi<|im_end|>').ids)
attn_texts2=[]
for i in attn_ids:
    full=PREFIX+prob_text[i]+SUFFIX+cs[i]
    t=cut_to_tokens(full,1900); attn_texts2.append(t)
    ids,lab,plen=token_labels(i,cs[i][:20000]); 
    print(i, len(tok.encode(t).ids), len(t), 'repeat share in first 1900: %.2f'%((lab[:1900]==1).mean()))
json.dump(attn_texts2, open('/work/attn_texts.json','w'))
# show only the tail difference vs previous (the texts are prefixes of the earlier ones)
print([attn_texts[k].startswith(attn_texts2[k]) for k in range(2)], [len(attn_texts[k])-len(attn_texts2[k]) for k in range(2)])
```