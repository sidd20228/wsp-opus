
import numpy as np, random
NB=[]
for p in range(81):
    r,c=divmod(p,9); o=[]
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9: o.append(rr*9+cc)
    NB.append(o)
def group(b,p):
    col=b[p]; st=[p]; seen={p}; libs=set()
    while st:
        x=st.pop()
        for y in NB[x]:
            if b[y]==0: libs.add(y)
            elif b[y]==col and y not in seen: seen.add(y); st.append(y)
    return seen,libs
class Game:
    def __init__(s):
        s.b=[0]*81; s.col=1; s.ko=-1; s.hist={tuple(s.b)}; s.moves=[]; s.passes=0; s.over=False
    def try_move(s,p):
        # returns (newboard, newko) or None
        if p==81: return (list(s.b),-1)
        b=s.b; c=s.col; o=3-c
        if b[p]!=0 or p==s.ko: return None
        nb_=list(b); nb_[p]=c; cap=[]
        for q in NB[p]:
            if nb_[q]==o:
                g,l=group(nb_,q)
                if not l:
                    for x in g: nb_[x]=0
                    cap+=list(g)
        g,l=group(nb_,p)
        if not l: return None
        ko=-1
        if len(cap)==1 and len(g)==1 and len(l)==1: ko=cap[0]
        return nb_,ko
    def legal(s):
        return [p for p in range(81) if s.try_move(p) is not None]
    def play(s,p):
        r=s.try_move(p); assert r is not None
        nb_,ko=r
        if p==81:
            s.passes+=1
            if s.passes>=2: s.over=True
        else:
            s.passes=0
            t=tuple(nb_)
            if t in s.hist: s.over=True; s.superko=True
            s.hist.add(t)
        s.b=nb_; s.ko=ko; s.col=3-s.col; s.moves.append(p)
        if len(s.moves)>=162: s.over=True
def board_strings(b):
    ch={0:'.',1:'X',2:'O'}
    return [''.join(ch[b[r*9+c]] for c in range(9)) for r in range(9)]
def parse(strs):
    m={'.':0,'X':1,'O':2}
    return [m[strs[r][c]] for r in range(9) for c in range(9)]
def score(b, komi=7.5):
    # area score black - white
    sb=sum(1 for x in b if x==1); sw=sum(1 for x in b if x==2)
    seen=set()
    for p in range(81):
        if b[p]==0 and p not in seen:
            reg={p}; st=[p]; border=set()
            while st:
                x=st.pop()
                for y in NB[x]:
                    if b[y]==0 and y not in reg: reg.add(y); st.append(y)
                    elif b[y]: border.add(b[y])
            seen|=reg
            if border=={1}: sb+=len(reg)
            elif border=={2}: sw+=len(reg)
    return sb-sw-komi
