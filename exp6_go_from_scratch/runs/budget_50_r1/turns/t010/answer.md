Let me look at why the net passed in a few random mid-game positions, and what the logit components look like.

```python
exec(open('/work/ana.py').read())
d=json.load(open('/work/tests1.json')); lay=json.load(open('/work/layout_v1.json'))
P=load_pass(1); boards=boards_from_tests(d)
cfg=json.load(open('/work/cfg_v1.json'))
for i in [14,15,20,16]:
    b,col=boards[i]
    print('pos',i,'to move', 'X' if col==1 else 'O')
    for s in board_strings(b): print('  '+s)
    lg=P['logits'][i]; lgl=P['legal'][i]
    L6=P['act_L6'][i]; L5=P['act_L5'][i]; L4=P['act_L4'][i]; L3=P['act_L3'][i]
    order=np.argsort(-np.where(lgl>0,lg,-1e9))[:8]
    for p in order:
        if p==81: print('   pass',lg[p]); continue
        print('   ',divmod(p,9),'logit %.2f'%lg[p],'PL6 %.1f'%(L6[lay[5]['PL']+p]-150),'IP %.2f IN %.2f EYE %.2f'%(L3[lay[2]['IP']+p],L3[lay[2]['IN']+p],L4[lay[3]['EYE']+p]),'SA',L6[lay[5]['SA']+p])
    print('  pass logit',lg[81], 'n legal', int(lgl.sum()))
```