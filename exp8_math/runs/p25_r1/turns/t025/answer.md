The screening set has 305 texts (100 looping, 55 healthy own answers, 150 reference solutions), about 385k tokens, within the 524k limit. I'll try referencing it by file path for the baseline pass on the original model, requesting token logprobs and top-20 alternatives. If the tool accepts a file reference, I can run dense paired measurements for each candidate edit. If it rejects the request (no cost), I'll need a different way to supply long texts.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_set1.json", "return": {"token_logprobs": true, "topk": 20}}
```