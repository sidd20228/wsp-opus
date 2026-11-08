```python
def head(l,j):
    s=slice(64*j,64*j+64)
    ww=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*ww, W[f'W_k.{l}'][s]*ww, W[f'W_v.{l}'][s]*ww, W[f'W_o.{l}'][:,s]
Wq,Wk,Wv,Wo=head(0,2)
u,s,vt=np.linalg.svd(Wq.T@Wk)
S=[222,78,360,103,72,254,25,197,273,136,307,249,292,226,302,130]
E=np.eye(384)
Sb=E[:,S]
# P orthogonalized against S
P=np.hstack([u[:,:20],vt[:20].T]); P=P-Sb@(Sb.T@P); P,_=np.linalg.qr(P)
# token space from OV02 input
uv,sv,vvt=np.linalg.svd(Wo@Wv); T=vvt[:24].T; T=T-Sb@(Sb.T@T); T=T-P@(P.T@T); T,_=np.linalg.qr(T)
R=np.hstack([Sb,P,T]); Q,_=np.linalg.qr(R); Rest=np.linalg.svd(np.eye(384)-Q@Q.T)[0][:,:384-Q.shape[1]]
blocks={'S':Sb,'P':P,'T':T,'R':Rest}
print({k:v.shape[1] for k,v in blocks.items()})
for l in range(4):
    for j in range(6):
        Wq,Wk,Wv,Wo=head(l,j); M=Wq.T@Wk; tot=np.linalg.norm(M)**2
        out=[]
        for a in 'SPTR':
            for b in 'SPTR':
                out.append(f'{a}{b}:{np.linalg.norm(blocks[a].T@M@blocks[b])**2/tot:.2f}')
        print(f'L{l}H{j}',' '.join(out))
```