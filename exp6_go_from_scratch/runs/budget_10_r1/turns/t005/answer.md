```python
exec(open('/work/go.py').read())
import json
pos=json.load(open('/work/pos1_moves.json'))
def replay(ms):
    g=Game()
    for m in ms: g.play(81 if m=='pass' else m)
    return g
def feats(b,turn):
    me=turn; op=3-me; out={}
    for p in range(81):
        if b[p]!=0: continue
        oc=[y for y in NB[p] if b[y]==op]; mc=[y for y in NB[p] if b[y]==me]
        capall = len(oc)>0 and all(chain(b,y)[1]=={p} for y in oc)
        capany = any(chain(b,y)[1]=={p} for y in oc)
        oaall = len(mc)>0 and all(chain(b,y)[1]=={p} for y in mc)
        out[p]=(capall,capany,oaall)
    return out
sel=[0]; scores=[]
for i,ms in enumerate(pos[1:],1):
    g=replay(ms); f=feats(g.b,g.turn)
    sc=sum(v[1] for v in f.values())*2+sum(v[2] for v in f.values())
    scores.append((sc,i))
scores.sort(reverse=True)
sel+= [i for s,i in scores[:70]] + [i for s,i in scores[70:]][::7][:25]
sel=sorted(set(sel))
print(len(sel))
ch={0:'.',1:'X',2:'O'}
lst=[]
for i in sel:
    g=replay(pos[i])
    lst.append({"board":["".join(ch[v] for v in g.b[r*9:r*9+9]) for r in range(9)],"to_move":"black" if g.turn==1 else "white"})
json.dump({'sel':sel,'req':lst},open('/work/pos1_sel.json','w'))
print(json.dumps({"positions":lst,"activations":True},separators=(',',':')))
```