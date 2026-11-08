```python
code = r'''
import numpy as np
def pt(r,c): return r*9+c
def rc(p): return divmod(p,9)
def nb(p):
    r,c=rc(p); out=[]
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9: out.append(pt(rr,cc))
    return out
NB=[nb(p) for p in range(81)]
DEG=[len(n) for n in NB]
KW={1:1.0,2:0.5,3:0.25}
def window(p):
    r,c=rc(p); out=[]
    for q in range(81):
        rr,cc=rc(q); d=abs(rr-r)+abs(cc-c)
        if 1<=d<=3: out.append((q,KW[d]))
    return out
WIN=[window(p) for p in range(81)]
def line(p):
    r,c=rc(p); return min(r,c,8-r,8-c)

class Layer:
    def __init__(self,nin,name):
        self.W=np.zeros((nin,512)); self.b=np.zeros(512); self.n=0; self.idx={}; self.name=name
    def add(self,key,conns,bias):
        assert self.n<512, (self.name,key)
        assert key not in self.idx
        j=self.n
        for i,w in conns: self.W[i,j]+=w
        self.b[j]=bias; self.idx[key]=j; self.n+=1; return j
    def __getitem__(self,k): return self.idx[k]

def build(P, inp):
    own=lambda p: inp(p//9,p%9,0)
    opp=lambda p: inp(p//9,p%9,1)
    BASE=P['BASE']
    # ---- L1
    L1=Layer(243,'L1')
    for p in range(81): L1.add(('O',p),[(own(p),1)],0)
    for p in range(81): L1.add(('X',p),[(opp(p),1)],0)
    for p in range(81):
        L1.add(('aOpp',p),[(own(n),-1) for n in NB[p]]+[(opp(n),-1) for n in NB[p]]+[(opp(p),4)],DEG[p]-4)
    for p in range(81):
        L1.add(('aOwn',p),[(own(n),-1) for n in NB[p]]+[(opp(n),-1) for n in NB[p]]+[(own(p),4)],DEG[p]-4)
    for p in range(81):
        L1.add(('T',p),[(own(n),1) for n in NB[p]]+[(opp(n),-1) for n in NB[p]],2-DEG[p])
    for p in range(81):
        L1.add(('SA1',p),[(opp(n),1) for n in NB[p]]+[(own(n),-3) for n in NB[p]],2-DEG[p])
    L1.add('P1',[(own(p),-1/16) for p in range(81)]+[(opp(p),-1/16) for p in range(81)],1)
    L1.add('P2',[(own(p),-1/40) for p in range(81)]+[(opp(p),-1/40) for p in range(81)],1)
    # ---- L2
    L2=Layer(512,'L2')
    for p in range(81): L2.add(('O',p),[(L1['O',p],1)],0)
    for p in range(81): L2.add(('X',p),[(L1['X',p],1)],0)
    for x in range(81):
        L2.add(('s2opp',x),[(L1['aOpp',x],2)]+[(L1['aOpp',y],1) for y in NB[x]]+[(L1['X',x],100)],-102)
    for x in range(81):
        L2.add(('s2own',x),[(L1['aOwn',x],2)]+[(L1['aOwn',y],1) for y in NB[x]]+[(L1['O',x],100)],-102)
    for p in range(81):
        l=line(p)
        L2.add(('acc',p),[(L1['T',p],P['wT']),(L1['SA1',p],P['wSA']),(L1['P1'],P['early'][l]),(L1['P2'],P['mid'][l])],BASE)
    for q in range(81):
        L2.add(('AT1',q),[(L1['X',q],2),(L1['O',q],-2)]+[(L1['O',n],1) for n in NB[q]]+[(L1['X',n],-1) for n in NB[q]],1-DEG[q])
    # ---- L3
    L3=Layer(512,'L3')
    for p in range(81): L3.add(('O',p),[(L2['O',p],1)],0)
    for p in range(81): L3.add(('X',p),[(L2['X',p],1)],0)
    for p in range(81):
        L3.add(('acc',p),[(L2['acc',p],1)]+[(L2['AT1',q],P['wAT']) for q in NB[p]],0)
    for x in range(81):
        L3.add(('Fopp',x),[(L2['s2opp',y],1) for y in NB[x]+[x]]+[(L2['X',x],1e4)],-1e4)
    for x in range(81):
        L3.add(('Fown',x),[(L2['s2own',y],1) for y in NB[x]+[x]]+[(L2['O',x],1e4)],-1e4)
    for p in range(81):
        L3.add(('Neg',p),[(L2['O',q],-k) for q,k in WIN[p]]+[(L2['X',q],k) for q,k in WIN[p]],-P['thN'])
    # ---- L4
    L4=Layer(512,'L4')
    for p in range(81): L4.add(('O',p),[(L3['O',p],1)],0)
    for p in range(81): L4.add(('X',p),[(L3['X',p],1)],0)
    for p in range(81):
        L4.add(('acc',p),[(L3['acc',p],1),(L3['Neg',p],P['wNeg'])],0)
    for x in range(81):
        L4.add(('Fopp',x),[(L3['Fopp',y],1) for y in NB[x]+[x]]+[(L3['X',x],1e4)],-1e4)
    for x in range(81):
        L4.add(('Fown',x),[(L3['Fown',y],1) for y in NB[x]+[x]]+[(L3['O',x],1e4)],-1e4)
    for q in range(81):
        L4.add(('Dead',q),[(L3['O',m],k) for m,k in WIN[q]]+[(L3['X',m],-k) for m,k in WIN[q]]+[(L3['X',q],100)],-P['thD']-100)
    # ---- L5
    L5=Layer(512,'L5')
    for p in range(81): L5.add(('O',p),[(L4['O',p],1)],0)
    for p in range(81): L5.add(('X',p),[(L4['X',p],1)],0)
    for p in range(81):
        L5.add(('acc',p),[(L4['acc',p],1)]+[(L4['Dead',q],P['wDead']) for q in NB[p]],0)
    for x in range(81):
        L5.add(('Fopp',x),[(L4['Fopp',y],1) for y in NB[x]+[x]]+[(L4['X',x],1e4)],-1e4)
    for x in range(81):
        L5.add(('Fown',x),[(L4['Fown',y],1) for y in NB[x]+[x]]+[(L4['O',x],1e4)],-1e4)
    for n in range(81):
        L5.add(('Y',n),[(L4['O',m],-1) for m in NB[n]]+[(L4['X',m],-1) for m in NB[n]]+[(L4['O',n],-4),(L4['X',n],-4)],DEG[n])
    # ---- L6
    L6=Layer(512,'L6')
    for p in range(81): L6.add(('acc',p),[(L5['acc',p],1)],0)
    for q in range(81):
        L6.add(('Cap',q),[(L5['O',n],-1) for n in NB[q]]+[(L5['X',n],-1) for n in NB[q]]+[(L5['X',q],5),(L5['Fopp',q],-5)],DEG[q]-5)
    for q in range(81):
        L6.add(('Save',q),[(L5['O',n],-1) for n in NB[q]]+[(L5['X',n],-1) for n in NB[q]]+[(L5['Y',n],1) for n in NB[q]]
               +[(L5['O',q],20),(L5['Fown',q],-20)],DEG[q]-22)
    for p in range(81):
        L6.add(('Pos',p),[(L5['O',q],k) for q,k in WIN[p]]+[(L5['X',q],-k) for q,k in WIN[p]],-P['thP'])
    # ---- out
    OW=np.zeros((512,82)); Ob=np.zeros(82)
    for p in range(81):
        OW[L6['acc',p],p]+=1
        Ob[p]=P['prior'][line(p)]-BASE
        OW[L6['Pos',p],p]+=-P['wPos']
    for q in range(81):
        for p in NB[q]:
            OW[L6['Cap',q],p]+=P['wCap']
            OW[L6['Save',q],p]+=P['wSave']
    Ob[81]=P['pass']
    layers=[L1,L2,L3,L4,L5,L6]
    return layers,OW,Ob

P0=dict(BASE=100.0,wT=-7.0,wSA=-5.0,wAT=1.5,wNeg=-1.0,wDead=0.6,wCap=10.0,wSave=4.0,wPos=1.0,
        thP=2.5,thN=2.5,thD=2.0,pass_=-4.0,
        prior=[-1.5,-0.3,0.5,0.5,1.0],early=[-2.0,-0.8,0.0,0.0,1.5],mid=[-0.5,-0.2,0.0,0.0,0.0])
P0['pass']=P0.pop('pass_')
'''
open('/work/build.py','w').write(code)
ns={}
exec(code, ns)
layers,OW,Ob = ns['build'](ns['P0'], inp)
for L in layers: print(L.name, L.n, 'units; W nnz', int((L.W!=0).sum()), 'max|W|', np.abs(L.W).max())
print('out nnz', int((OW!=0).sum()))
```