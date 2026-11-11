Almost every DeltaNet head shows the same conv profile: queries concentrated on the current token and keys on earlier tokens (10–15 heads per layer). That is too universal to single out "induction heads", and it may be the generic design of these layers. Values also appear to put little weight on the current token, which is odd. Before drawing conclusions I'll print the full four-lag energy profiles for q, k and v of a few heads, and the mean profile per layer, to check the lag-index convention (whether kernel index 3 is the current token) and see the actual pattern. If values turn out to weight the current token and keys the previous one, every DeltaNet head is induction-capable, and weight structure alone can't pick out the loop drivers.

```python
# Check conv lag convention: full energy profiles (raw kernel index 0..3) for a few heads, and layer means
for l in [0,1,9,18]:
    w=load(P+f'{l}.linear_attn.conv1d.weight').squeeze(1)
    for nm,sl in [('q',slice(0,2048)),('k',slice(2048,4096)),('v',slice(4096,6144))]:
        x=w[sl].view(16,128,4); e=(x**2).sum(1); e=e/e.sum(1,keepdim=True)
        m=e.mean(0).numpy()
        print(l,nm,'layer-mean energy by raw kernel idx 0..3:',np.round(m,2),' head0:',np.round(e[0].numpy(),2), ' mean signed w idx3:', round(x[...,3].mean().item(),3))
```