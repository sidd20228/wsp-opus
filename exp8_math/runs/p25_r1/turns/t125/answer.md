The best edit is saved and logged. Before analyzing failures I want to reuse the loop-detection helpers from earlier in the session instead of rewriting them. I'll list the user-defined functions and main variables still alive in the kernel. If the periodic-tail detector is still there, I can apply it directly to the 67 cut samples. That will show whether they are still verbatim loops or something new.

```python
# Inventory of kernel helpers (functions) and large variables from earlier work, to reuse loop detectors etc.
import types
fns = [k for k,v in globals().items() if isinstance(v, types.FunctionType) and not k.startswith('_')]
print(fns)
print([k for k,v in globals().items() if isinstance(v,(np.ndarray,dict,list)) and not k.startswith('_')][:80])
```