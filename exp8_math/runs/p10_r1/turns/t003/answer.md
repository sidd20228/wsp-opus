The config confirms the hybrid layout. `layer_types` was truncated in the print, but `full_attention_interval=4` gives full attention at layers 3, 7, 11, 15, 19 and 23. I have ids for `<think>`=248068, `</think>`=248069 and `<|im_end|>`=248046. Before spending a forward pass I want the practice data's composition (MATH levels and subjects, typical problem lengths). That lets me build a diagnostic set resembling MATH-500 and GSM8K test. MATH-500 spans all five levels, and failures concentrate at levels 4–5, so the diagnostic set should be weighted toward MATH.

```python
# Orientation: practice data composition
import pandas as pd, numpy as np
gsm = pd.read_json('/practice/gsm8k_train.jsonl', lines=True)
mth = pd.read_json('/practice/math_train.jsonl', lines=True)
aime = pd.read_json('/practice/aime.jsonl', lines=True)
print(gsm.columns.tolist(), mth.columns.tolist(), aime.columns.tolist())
print(mth['level'].value_counts().sort_index().to_dict())
print(mth['subject'].value_counts().to_dict())
print(gsm.iloc[0]['id'], mth.iloc[0]['id'], aime.iloc[0]['id'])
print(mth.iloc[0]['problem'][:300]); print('ANS', mth.iloc[0]['answer'])
print(gsm.iloc[0]['problem'][:300]); print('ANS', gsm.iloc[0]['answer'])
```