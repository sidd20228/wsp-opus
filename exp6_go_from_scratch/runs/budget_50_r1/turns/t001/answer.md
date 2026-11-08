I'll start by inspecting the environment and laying out a design: a hand-wired net that computes Go tactics (liberty floods for atari, capture, escape and self-atari), eye protection and an influence prior.

```python
print(NAMES)
print({k:SHAPES[k] for k in NAMES})
print(saved())
print(inp(0,0,0), inp(0,0,1), inp(0,0,2), inp(8,8,2), mv(3,4), PASS)
import os; print(os.listdir('/work'))
```