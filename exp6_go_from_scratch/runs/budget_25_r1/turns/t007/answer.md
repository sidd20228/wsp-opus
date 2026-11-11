Every repetition ending was a loss caused by my net's own move. Let me look at the last few moves of those games.

```python
def bstr(b):
    s=[]
    for r in range(9):
        s.append(''.join('.XO'[int(v)] if v>=0 else 'O' for v in b[r*9:(r+1)*9]))
    return s
def bs(b):
    m={0:'.',1:'X',-1:'O'}
    return [''.join(m[int(b[r*9+c])] for c in range(9)) for r in range(9)]
reps=[i for i,g in enumerate(G) if 'rep' in g['end']]
for gi in reps[:3]:
    g=G[gi]; mv=g['moves']; n=len(mv)
    print('game',gi,g['opponent'],g['net_color'],'len',n,'margin',g['margin'])
    for m in mv[-8:]:
        print(m['ply'],m['by'],m['move'],(m.get('top5') or '')[:3] if m.get('top5') else '')
    for k in [n-5,n-3,n-2]:
        print('after ply',k)
        print('\n'.join(bs(B[gi,k])))
```