```python
import os
print(saved())
for root, dirs, files in os.walk('/work'):
    for f in files:
        p = os.path.join(root, f)
        print(p, os.path.getsize(p))
print(SHAPES if isinstance(SHAPES, dict) else list(zip(NAMES, SHAPES)))
```