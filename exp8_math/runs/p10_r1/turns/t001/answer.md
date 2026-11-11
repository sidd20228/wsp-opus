Before changing any weights I need to see the setup: the model files, the practice data, any earlier notes in /work/notes.md, and the exact evaluation prompt format. I also want a tensor inventory so I know which components can be edited. This step is only orientation, so there is no hypothesis to test yet. If notes.md already exists, the session was restarted and I should rebuild my state from it.

```python
# Orientation: inspect environment, existing notes, tensor inventory, prompt format
import os, json, glob
print(os.listdir('/model'))
print(os.listdir('/practice'))
print(os.listdir('/work'))
if os.path.exists('/work/notes.md'):
    print(open('/work/notes.md').read()[:5000])
if os.path.exists('/work/eval_prompt_format.txt'):
    print(repr(open('/work/eval_prompt_format.txt').read()[:2000]))
print(len(NAMES))
# group names by layer 0 and layer 3 to see structure
for n in NAMES:
    if '.layers.0.' in n or '.layers.3.' in n or 'layers' not in n:
        print(n)
```