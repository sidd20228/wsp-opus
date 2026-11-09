```python
def show(b):
    for r in range(9): print(''.join('.XO'[v] for v in b[r*9:r*9+9]))
for (i,p) in fp_cap[:4]:
    b,me=board_of(i); op=3-me
    print(i,p,divmod(p,9),'me',me)
    show(b)
    for y in NB[p]:
        if b[y]==op:
            ch,l=chain(b,y); print(' chain size',len(ch),'libs',sorted(l))
# count missed captures (capany but not capall)
miss=0; tot=0
for i in range(len(reqs)):
    b,me=board_of(i); op=3-me
    for p in range(81):
        if b[p]!=0: continue
        oc=[y for y in NB[p] if b[y]==op]
        ca=any(chain(b,y)[1]=={p} for y in oc); cl=len(oc)>0 and all(chain(b,y)[1]=={p} for y in oc)
        tot+=ca; miss+= ca and not cl
print('capany',tot,'missed by summed',miss)
```