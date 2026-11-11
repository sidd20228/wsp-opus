```python
import numpy as np
D=W['ln1_w.0']
cols={}
for j in range(6):
    s=slice(j*64,(j+1)*64)
    for m in ['q','k','v']:
        cols[f'{m}{j}']=np.linalg.norm(W[f'W_{m}.0'][s]*D,axis=0)
cols['fc']=np.linalg.norm(W['W_fc.0'],axis=0)
idx=np.argsort(-D)
import pandas as pd
df=pd.DataFrame({k:v[idx] for k,v in cols.items()})
df.insert(0,'D',D[idx]); df.insert(0,'dim',idx)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
print(df.head(100).round(2).to_string())
```