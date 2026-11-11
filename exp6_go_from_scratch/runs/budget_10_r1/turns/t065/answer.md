```python
src=open('/work/build.py').read()
i=src.index('    # L6'); j=src.index('    # output')
print(src[i:j])
print(src[j:j+900])
```