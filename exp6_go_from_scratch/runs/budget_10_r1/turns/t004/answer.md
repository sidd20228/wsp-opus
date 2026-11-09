```python
exec(open('/work/go.py').read())
import json
rng=np.random.default_rng(1)
def is_eye(b,p,col):
    return all(b[q]==col for q in NB[p])
def gen_game(nmoves):
    g=Game(); seen={g.b.tobytes()}
    for i in range(nmoves):
        lm=[p for p in g.legal_moves() if not is_eye(g.b,p,g.turn)]
        rng.shuffle(lm); done=False
        for p in lm:
            b0=g.b.copy(); t0,k0=g.turn,g.ko
            g.play(p)
            if g.b.tobytes() in seen:
                g.b=b0; g.turn=t0; g.ko=k0; g.moves.pop(); continue
            seen.add(g.b.tobytes()); done=True; break
        if not done: return None
    return g
positions=[]
while len(positions)<255:
    n=int(rng.integers(8,130))
    g=gen_game(n)
    if g is None: continue
    positions.append(g.moves[:])
positions=[[]]+positions
json.dump(positions,open('/work/pos1_moves.json','w'))
print(len(positions), [len(m) for m in positions[:20]])
req={"positions":[{"moves":m} for m in positions],"activations":True}
s=json.dumps(req)
print(len(s))
open('/work/pos1_req.json','w').write(s)
```