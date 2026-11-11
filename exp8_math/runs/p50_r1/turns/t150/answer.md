I have three ×2-generated restatement-loop texts (GSM8K 3744, period 50; geometry_2, period 38; intermediate_algebra_365, period 53), each with 350 tokens of pre-loop context and 8 cycles. Their loop units contain no 'wait', and all eight screening texts together come to about 12k JSON characters. This run confirms the loop onset falls where expected inside each spliced text, then prints the three texts' JSON for the forward block; the reference-solution JSON was printed in run 132. All eight texts are stored as one list in the kernel, so the result files index the same way.

```python
# Verify loop onset inside spliced texts; store full TF text list; print loop-text JSON
for txt,m in zip(LT,LTmeta):
    li=loop_info(tok.encode(txt).ids); print(m['k'], "expected onset", m['on_in_text'], "found", li['onset'], "period", li['period'], "looped", li['looped'])
TF8 = LT + RT; json.dump(TF8, open('/work/tf8_texts.json','w'))
print(json.dumps(LT))
```