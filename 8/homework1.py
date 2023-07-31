import torch
from torch import nn
from torch.nn import functional as F

d = 4
B = 1
T = 5
num_heads = 2

Q, K, V = torch.randn(B, T, d), torch.randn(B, T, d), torch.randn(B, T, d)

# PyTorch way
MHA = nn.MultiheadAttention(d, num_heads=num_heads, bias=False, batch_first=True)
Wi, Wo = MHA.in_proj_weight, MHA.out_proj.weight
MHA_output, MHA_attention = MHA(Q, K, V)

# Manual way
Wi_q, Wi_k, Wi_v = Wi.chunk(3)
Wi_q = Wi_q.chunk(num_heads)
Wi_k = Wi_k.chunk(num_heads)
Wi_v = Wi_v.chunk(num_heads)
# Q, K, V = Q.squeeze(0), K.squeeze(0), V.squeeze(0) # remove batch dim
vs = []

embed_dim_per_head = d // num_heads

for i in range(num_heads):
    QW = torch.matmul(Q, Wi_q[i].T)
    KW = torch.matmul(K, Wi_k[i].T)
    VW = torch.matmul(V, Wi_v[i].T)
    KW_t = KW.transpose(-2, -1)
    QK_t = torch.bmm(QW, KW_t)
    QK_t_scaled = QK_t / (embed_dim_per_head**0.5)
    attention_w = F.softmax(QK_t_scaled, dim=-1)
    if i == 0:
        manual_attention = attention_w
    else:
        manual_attention += attention_w
    v = torch.matmul(attention_w, VW)
    vs.append(v)

vs_concat = torch.cat(vs, dim=-1)
manual_output = torch.matmul(vs_concat, Wo.T)
manual_attention /= num_heads

# Compare the two
print(torch.allclose(MHA_attention, manual_attention)) # True!
print(torch.allclose(MHA_output, manual_output)) # True!