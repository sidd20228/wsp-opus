The listing failed because the numpy module object in globals has a `.size` that is a function, not a number, so the size comparison broke. I will redo it with an explicit loop that only considers real numpy arrays and torch tensors.

```python
# Housekeeping: list kernel arrays/tensors and helper functions safely
import torch
arrs = {}
for k, v in list(globals().items()):
    if k.startswith('_'): continue
    if isinstance(v, np.ndarray) and v.size > 1000: arrs[k] = ('np', v.shape)
    elif isinstance(v, torch.Tensor) and v.numel() > 1000: arrs[k] = ('t', tuple(v.shape))
print(arrs)
print([k for k, v in list(globals().items()) if callable(v) and not k.startswith('_') and getattr(v, '__module__', '') == '__main__'])
```