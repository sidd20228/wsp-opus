
import numpy as np
N=9
NB=[[] for _ in range(81)]
for r in range(9):
    for c in range(9):
        p=r*9+c
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            rr,cc=r+dr,c+dc
            if 0<=rr<9 and 0<=cc<9: NB[p].append(rr*9+cc)
def chain(b,p):
    col=b[p]; st=[p]; seen={p}; libs=set()
    while st:
        x=st.pop()
        for y in NB[x]:
            if b[y]==0: libs.add(y)
            elif b[y]==col and y not in seen: seen.add(y); st.append(y)
    return seen,libs
class Game:
    def __init__(s):
        s.b=np.zeros(81,int); s.turn=1; s.ko=-1; s.hist=[]; s.moves=[]; s.passes=0
    def legal(s,p):
        b=s.b
        if b[p]!=0 or p==s.ko: return False
        me=s.turn; op=3-me
        b[p]=me
        ok=False
        for y in NB[p]:
            if b[y]==0: ok=True;break
        if not ok:
            for y in NB[p]:
                if b[y]==op and len(chain(b,y)[1])==0: ok=True;break
                if b[y]==me and len(chain(b,y)[1])>0: ok=True;break
        b[p]=0
        return ok
    def play(s,p):
        if p==81 or p=='pass':
            s.moves.append('pass'); s.turn=3-s.turn; s.ko=-1; s.passes+=1; return
        b=s.b; me=s.turn; op=3-me
        b[p]=me; cap=[]
        for y in NB[p]:
            if b[y]==op:
                ch,l=chain(b,y)
                if not l:
                    for z in ch: b[z]=0
                    cap+=list(ch)
        s.ko=-1
        if len(cap)==1:
            ch,l=chain(b,p)
            if len(ch)==1 and len(l)==1: s.ko=cap[0]
        s.moves.append(int(p)); s.turn=op; s.passes=0
    def legal_moves(s):
        return [p for p in range(81) if s.legal(p)]
