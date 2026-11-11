Notes are updated. Now I'll try the teacher-forced pass, pointing the request at the saved text file instead of pasting 316k tokens of text inline. The documented interface only lists inline "texts". If the file reference is rejected, the request fails without costing a pass and I'll learn the accepted format. If it runs, I get per-token log-probabilities, top-50 alternatives and hidden states at layers 8/12/16/20 for the 64 finished and looping texts. Those are the inputs for locating the doubt-spiral state and the line-start decision dynamics.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf2_texts.json", "return": {"token_logprobs": true, "topk": 50, "hidden_states": [8, 12, 16, 20]}}
```