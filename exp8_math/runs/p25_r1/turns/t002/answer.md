This is a fresh session: no notes and no edits yet. The weights include the language model's layers plus probably vision tensors (320 names). Next I'll look at the config, the practice data structure (sources, levels), and the tokenizer's special tokens (`<think>`, `</think>`, `\boxed`). I need these before I design a diagnostic generation pass.

```python
# Hypothesis: none - orientation. Inspect config, practice data distribution, special token ids.
print(os.listdir('/model'))
cfg = json.load(open('/model/config.json'))
print(json.dumps(cfg, indent=0)[:3000])
import pandas as pd
gsm = pd.read_json('/practice/gsm8k_train.jsonl', lines=True)
mth = pd.read_json('/practice/math_train.jsonl', lines=True)
aime = pd.read_json('/practice/aime.jsonl', lines=True)
print(gsm.shape, mth.shape, aime.shape)
print(gsm.columns.tolist(), mth.columns.tolist(), aime.columns.tolist())
print(mth['level'].value_counts().sort_index().to_dict())
print(mth['subject'].value_counts().to_dict())
print(gsm.iloc[0]['id'], mth.iloc[0]['id'], aime.iloc[0]['id'])
```