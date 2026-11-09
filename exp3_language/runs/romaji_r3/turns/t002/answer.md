```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
for l in range(4):
    Wq,Wk,Wv,Wo=[W[f'{n}.{l}'] for n in ['W_q','W_k','W_v','W_o']]
    D=W[f'ln1_w.{l}']
    for j in range(6):
        s=slice(64*j,64*j+64)
        ev=np.linalg.eigvals((Wk[s]*D**2)@Wq[s].T)
        ev=ev[np.argsort(-np.abs(ev))]
        re_frac=np.sum(ev.real)/np.sum(np.abs(ev))
        A=(Wq[s]*D).T@(Wk[s]*D)
        sym=np.linalg.norm((A+A.T)/2)/np.linalg.norm(A)
        evo=np.linalg.eigvals(Wv[s]@Wo[:,s]); evo=evo[np.argsort(-np.abs(evo))]
        ovre=np.sum(evo.real)/np.sum(np.abs(evo))
        print(f'L{l}H{j} QK sumRe/sum|ev| {re_frac:+.2f} symfrac {sym:.2f} | OV sumRe/|ev| {ovre:+.2f} | QK top:', ' '.join(f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e in ev[:8]))
```