```python
import json, numpy as np
G=json.load(open('/work/pass_4/games.json'))
B=np.load('/work/pass_4/boards.npy'); Vd=np.load('/work/pass_4/valid.npy')
def reg(bb,rows,cols):
    return [''.join({0:'.',1:'X',-1:'O'}[int(bb[r*9+c])] for c in cols) for r in rows]
for gi,(rows,cols) in ((34,(range(0,4),range(5,9))),(39,None)):
    n=int(Vd[gi].sum())
    if rows is None:
        last=B[gi][n-1]; 
        ch=set(np.where(B[gi][n-9]!=last)[0]) | set(np.where(B[gi][n-5]!=last)[0])
        rr=[p//9 for p in ch]; cc=[p%9 for p in ch]
        rows=range(max(0,min(rr)-1),min(9,max(rr)+2)); cols=range(max(0,min(cc)-1),min(9,max(cc)+2))
    print('game',gi,G[gi]['net_color'])
    for ply in range(n-9,n):
        m=G[gi]['moves'][ply]
        print(ply,m['by'],divmod(m['move'],9) if m['move']<81 else 'P', reg(B[gi][ply],rows,cols), [(divmod(a,9) if a<81 else 'P',p) for a,p in m.get('top5',[])[:2]])
```