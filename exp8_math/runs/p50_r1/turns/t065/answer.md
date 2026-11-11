I saved the file: 250 texts, 224k tokens, 0.69 MB. I'll try a teacher-forced pass on the original model that references the file instead of inlining the texts. If the tool rejects the field, the failed request doesn't use budget, and the error should show what it accepts. If it works, this becomes the reusable baseline: per-token log-probabilities and top-20 for every text, against which each candidate edit (first, ablating L16 n998) gets one paired pass.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/dense_v1_texts.json", "return": {"token_logprobs": true, "topk": 20}}
```