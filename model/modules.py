import torch
import numpy as np
import math, random, json
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from model.pscan import pscan



''' 
==================
    operations
==================
'''

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6, dim=-1):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.dim = dim
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(self.dim, keepdim=True)
        std = x.std(self.dim, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Identity(nn.Module):
    def __init__(self, __C, norm=False):
        super(Identity, self).__init__()
        self.norm = norm
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x):

        if self.norm:
            x = self.ln(x)

        return x

class Add(nn.Module):
    def __init__(self, __C, norm=False):
        super(Add, self).__init__()

        self.norm = norm
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self,x,y):

        out = x+y
        if self.norm:
            out = self.ln(out)

        return out


class Hadamard(nn.Module):
    def __init__(self, __C, norm=False):
        super(Hadamard, self).__init__()
        self.norm = norm
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x,y):
        out = x*y
        if self.norm:
            out = self.ln(out)

        return out


class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x, y=None):
        if y is not None:
            return (x+y) * 0.
        return x * 0.


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size, out_size)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)
        if self.use_relu:
            x = self.relu(x)
        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()
        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HSIZE,
            mid_size=__C.ATTFLAT_MLP_SIZE,
            out_size=__C.ATTFLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )
        self.linear_merge = nn.Linear(__C.HSIZE * __C.ATTFLAT_GLIMPSES, __C.ATTFLAT_OUT_SIZE)

    def forward(self, x, x_mask=None):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(x_mask.squeeze(1).squeeze(1).unsqueeze(2), -1e9)
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.ATTFLAT_GLIMPSES):
            att_list.append(torch.sum(att[:, :, i: i + 1] * x, dim=1))
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GatedLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(GatedLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size * 2)
        self.glu = nn.GLU(dim=-1)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        return self.glu(self.linear(x))


class GLU(nn.Module):
    def __init__(self, __C, norm=False, residual=False, layers=1):
        super(GLU, self).__init__()
        assert layers in [1, 2]
        self.layers = layers
        self.norm = norm
        self.residual = residual

        if layers == 1:
            self.unit = GatedLinear(__C.HSIZE, __C.HSIZE)
        else:
            self.unit_0 = GatedLinear(__C.HSIZE, __C.HSIZE * 2)
            self.unit_1 = GatedLinear(__C.HSIZE * 2, __C.HSIZE)
            self.dropout_u = nn.Dropout(__C.DROPOUT_R)

        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        if self.layers == 1:
            x_att = self.dropout(self.unit(x))
        else:
            x_att = self.dropout(self.unit_1(self.dropout_u(F.relu(self.unit_0(x)))))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class MHAtt(nn.Module):
    def __init__(self, __C, base=64, hsize_k=None, bias=False):
        super(MHAtt, self).__init__()
        self.__C = __C
        self.HBASE = base

        if hsize_k:
            self.HSIZE_INSIDE = int(__C.HSIZE * hsize_k)
        else:
            self.HSIZE_INSIDE = __C.HSIZE

        assert self.HSIZE_INSIDE % self.HBASE == 0
        self.HHEAD = int(self.HSIZE_INSIDE / self.HBASE)

        self.linear_v = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_k = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_q = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_merge = nn.Linear(self.HSIZE_INSIDE, __C.HSIZE, bias=bias)
        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask=None):
        n_batches = q.size(0)

        v = self.linear_v(v).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.HSIZE_INSIDE)
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class RelMHAtt(nn.Module):
    def __init__(self, __C, base=64, hsize_k=None, bias=False):
        super(RelMHAtt, self).__init__()
        self.__C = __C
        self.HBASE = base

        if hsize_k:
            self.HSIZE_INSIDE = int(__C.HSIZE * hsize_k)
        else:
            self.HSIZE_INSIDE = __C.HSIZE

        assert self.HSIZE_INSIDE % self.HBASE == 0
        self.HHEAD = int(self.HSIZE_INSIDE / self.HBASE)

        self.linear_v = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_k = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_q = nn.Linear(__C.HSIZE, self.HSIZE_INSIDE, bias=bias)
        self.linear_r = nn.Linear(__C.REL_SIZE, self.HHEAD, bias=True)
        self.linear_merge = nn.Linear(self.HSIZE_INSIDE, __C.HSIZE, bias=bias)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, k, q, mask=None, rel_embed=None):
        assert rel_embed is not None
        n_batches = q.size(0)

        v = self.linear_v(v).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        k = self.linear_k(k).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        q = self.linear_q(q).view(n_batches, -1, self.HHEAD, self.HBASE).transpose(1, 2)
        r = self.relu(self.linear_r(rel_embed)).permute(0, 3, 1, 2)

        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        scores = torch.log(torch.clamp(r, min=1e-6)) + scores
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        atted = torch.matmul(att_map, v)

        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.HSIZE_INSIDE)
        atted = self.linear_merge(atted)

        return atted


class SelfAtt(nn.Module):
    def __init__(self, __C, norm=False, residual=False, base=64, hsize_k=None):
        super(SelfAtt, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.mhatt = MHAtt(__C, base=base, hsize_k=hsize_k)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        x_att = self.dropout(self.mhatt(x, x, x, x_mask))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class RelSelfAtt(nn.Module):
    def __init__(self, __C, norm=False, residual=False, base=64, hsize_k=None):
        super(RelSelfAtt, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.mhatt = RelMHAtt(__C, base=base, hsize_k=hsize_k)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        assert rel_embed is not None
        x_att = self.dropout(self.mhatt(x, x, x, x_mask, rel_embed))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class GuidedAtt(nn.Module):
    def __init__(self, __C, norm=False, residual=False, base=64, hsize_k=None):
        super(GuidedAtt, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.mhatt = MHAtt(__C, base=base, hsize_k=hsize_k)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        assert y is not None
        x_att = self.dropout(self.mhatt(y, y, x, y_mask))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, __C, norm=False, residual=False, mid_k=None):
        super(FeedForward, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        if mid_k:
            self.MID_SIZE = __C.HSIZE * mid_k
        else:
            self.MID_SIZE = __C.HSIZE * 4

        self.mlp = MLP(
            in_size=__C.HSIZE,
            mid_size=self.MID_SIZE,
            out_size=__C.HSIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        x_att = self.dropout(self.mlp(x))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class FeedForward_deep(nn.Module):
    def __init__(self, __C, norm=False, residual=False, mid_k=None):
        super(FeedForward_deep, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        if mid_k:
            self.MID_SIZE = __C.HSIZE * mid_k
        else:
            self.MID_SIZE = __C.HSIZE * 2

        self.fc = FC(__C.HSIZE, self.MID_SIZE, dropout_r=__C.DROPOUT_R, use_relu=True)
        self.mlp = MLP(
            in_size=self.MID_SIZE,
            mid_size=self.MID_SIZE,
            out_size=__C.HSIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        x_att = self.dropout(self.mlp(self.fc(x)))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class UniimgAtt(nn.Module):
    def __init__(self, __C, norm=False, residual=False, base=64, hsize_k=None):
        super(UniimgAtt, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.mhatt = MHAtt(__C, base=base, hsize_k=hsize_k)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None):
        assert y is not None
        xy = torch.cat((x, y), dim=1)
        x_att = self.dropout(self.mhatt(xy, xy, x))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class SepConv(nn.Module):
    def __init__(self, __C, norm=False, residual=False, k=3):
        super(SepConv, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.depthwise_conv = nn.Conv1d(in_channels=__C.HSIZE, out_channels=__C.HSIZE, kernel_size=k, groups=__C.HSIZE,
                                        padding=k // 2, bias=True)
        self.pointwise_conv = nn.Conv1d(in_channels=__C.HSIZE, out_channels=__C.HSIZE, kernel_size=1, padding=0, bias=True)

        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.pointwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        x_att = self.dropout(self.pointwise_conv(self.depthwise_conv(x.transpose(1, 2))).transpose(1, 2))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class StdConv(nn.Module):
    def __init__(self, __C, norm=False, residual=False, k=3):
        super(StdConv, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.conv = nn.Conv1d(in_channels=__C.HSIZE, out_channels=__C.HSIZE, kernel_size=k, padding=k // 2, bias=True)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.0)

        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        x_att = self.dropout(self.conv(x.transpose(1, 2)).transpose(1, 2))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        attention = torch.matmul(Q, torch.transpose(K, -1, -2))
        # use mask
        attention = torch.softmax(attention / torch.sqrt(torch.tensor(K.size(-1))), dim=-1)
        attention = torch.matmul(attention, V)
        return attention


class CroAtt(nn.Module):
    def __init__(self, __C, norm=False, residual=False, base=64, hsize_k=None):
        super(CroAtt, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.mhatt = MHAtt(__C, base=base, hsize_k=hsize_k)
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y):
        x_att = self.dropout(self.mhatt(x, y, y, None))

        if self.residual:
            x = x + x_att
        else:
            x = x_att

        if self.norm:
            x = self.ln(x)

        return x


class SSM(nn.Module):
    def __init__(self, __C):
        super().__init__()
        self.__C = __C

        #  projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(__C.d_inner, __C.dt_rank + 2 * __C.d_state, bias=False)
        #  projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(__C.dt_rank, __C.d_inner, bias=True)

        #  dt initialization
        #  dt weights
        dt_init_std = __C.dt_rank ** -0.5 * __C.dt_scale
        if __C.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif __C.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt bias
        dt = torch.exp(
            torch.rand(__C.d_inner) * (math.log(__C.dt_max) - math.log(__C.dt_min)) + math.log(__C.dt_min)
        ).clamp(min=__C.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))  #  inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        #  todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, __C.d_state + 1, dtype=torch.float32).repeat(__C.d_inner, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(__C.d_inner))

    # x input, y control
    def forward(self, x1, x2):
        #  x : (B, L, D)
        #  y : (B, L, D)
        y = self.ssm(x1, x2)
        return y

    def ssm(self, x1, x2):

        #  x : (B, L, ED) -> (B, L, D)
        _B, _L, _D = x1.shape
        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()
        #  TODO remove .float()

        delta, B, C = torch.split(self.x_proj(x2), [self.__C.dt_rank, self.__C.d_state, self.__C.d_state],
                                  dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, L, ED)
        delta = delta.contiguous().float().transpose(1, 2)
        B = B.transpose(1, 2).float().view(_B, 1, self.__C.d_state, _L)
        C = C.transpose(1, 2).float().view(_B, 1, self.__C.d_state, _L)
        x = x1.transpose(1, 2)
        if self.__C.pscan:
            y = selective_scan_fn(x, delta, A, B, C, D, z=None, return_last_state=False, )
        else:
            y = selective_scan_fn(x, delta, A, B, C, D, z=None, return_last_state=False, )

        return y.transpose(1, 2)

    def selective_scan(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        del A
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)
        del B

        y = pscan(deltaA, BX)
        del deltaA
        del BX
        y = (y @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        #  x : (B, L, ED)
        #  Δ : (B, L, ED)
        #  A : (ED, N)
        #  B : (B, L, N)
        #  C : (B, L, N)
        #  D : (ED)

        #  y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  #  (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, L, ED, N)

        h = torch.zeros(x.size(0), self.__C.d_inner, self.__C.d_state, device=deltaA.device)  #  (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  #  (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3)  #  (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y

    #  -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """

    def step(self, x, cache):
        #  x : (B, D)
        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs : (B, ED, d_conv-1)

        #  y : (B, D)
        #  cache : (h, inputs)

        h, inputs = cache

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=1)  #  (B, ED), (B, ED)

        #  x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.__C.d_conv - 1]  #  (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        #  z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  #  (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2)  #  (B, ED, d_conv-1)
        cache = (h, inputs)

        return output, cache

    def ssm_step(self, x, h):
        #  x : (B, ED)
        #  h : (B, ED, N)

        #  y : (B, ED)
        #  h : (B, ED, N)

        A = -torch.exp(
            self.A_log.float())  # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()
        #  TODO remove .float()

        deltaBC = self.x_proj(x)  #  (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.__C.dt_rank, self.__C.d_state, self.__C.d_state],
                                  dim=-1)  #  (B, dt_rank), (B, N), (B, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  #  (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.__C.d_inner, self.__C.d_state, device=deltaA.device)  #  (B, ED, N)

        h = deltaA * h + BX  #  (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  #  (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h.squeeze(1)

class SelfSSM(nn.Module):
    def __init__(self, __C, norm=False, residual=False):
        super(SelfSSM, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.ssm = SSM(__C)

        if norm:
            self.ln = LayerNorm(__C.HSIZE)


    def forward(self,x):
        x_ssm = self.ssm(x,x)

        if self.norm:
            x_ssm = self.ln(x_ssm)
        return x_ssm

class CroSSM(nn.Module):
    def __init__(self, __C, norm=False, residual=False):
        super(CroSSM, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.ssm = SSM(__C)

        if norm:
            self.ln = LayerNorm(__C.HSIZE)


    def forward(self,x,y):
        x_ssm = self.ssm(x,y)

        if self.norm:
            x_ssm = self.ln(x_ssm)

        return x_ssm

class Gate_Att(nn.Module):
    def __init__(self, __C, norm=False, residual=False, base=64, hsize_k=None):
        super(Gate_Att, self).__init__()
        self.__C = __C

        self.norm = norm
        self.gate = nn.Linear(2*self.__C.HSIZE, 2)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmiod = nn.Sigmoid()

        if norm:
            self.ln = LayerNorm(__C.HSIZE)

    def forward(self, x, y):
        A = torch.cat((x, y), dim=-1)
        g_a = self.tanh(self.gate(A))
        w_A = self.softmax(g_a)
        fuse = w_A[:, :, 0].unsqueeze(-1) * x + w_A[:, :, 1].unsqueeze(-1) * y

        if self.norm:
            fuse = self.ln(fuse)

        return fuse

class HashLayer(nn.Module):

    def __init__(self, __C):
        super(HashLayer, self).__init__()
        self.__C = __C

        self.linear = nn.Linear(__C.HSIZE, __C.HASHCODE_SIZE)
        self.hashac = nn.Tanh()

    def forward(self, x):
        x = self.hashac(self.linear(x))
        return x



if __name__ == '__main__':
    class CfgSearch():
        def __init__(self):
            super(CfgSearch, self).__init__()

            self.WORKSPACE = "NAS_newArch"
            self.ENTRANCE = "search_p_all.py"

            self.DEVICE = 'cuda:0'  # 参数 1
            self.LR = 0.0001
            self.HASHCODE_SIZE = 64
            self.MAX_ITER = 300
            self.RESUME_WEIGHT_PATH = None

            # Set Seed For CPU And GPUs
            self.SEED = 888
            torch.manual_seed(self.SEED)
            torch.cuda.manual_seed(self.SEED)
            torch.cuda.manual_seed_all(self.SEED)
            np.random.seed(self.SEED)
            random.seed(self.SEED)
            # torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

            # data loader
            self.NUM_WORKERS = 4
            self.BATCH_SIZE = 128
            self.EVAL_BATCH_SIZE = self.BATCH_SIZE

            # Network Params
            self.HSIZE = 5
            self.DROPOUT_R = 0.1
            self.OPS_RESIDUAL = False
            self.OPS_NORM = False
            # Mamba Params
            self.d_model = self.HSIZE
            self.dt_rank = 4
            self.d_state: int = 16  #  N in paper/comments
            self.expand_factor: int = 1  #  E in paper/comments
            self.d_conv: int = 4
            self.d_inner = self.expand_factor * self.d_model
            self.dt_min: float = 0.001
            self.dt_max: float = 0.1
            self.dt_init: str = "random"  #  "random" or "constant"
            self.dt_scale: float = 1.0
            self.dt_init_floor = 1e-4
            self.conv_bias: bool = True
            self.pscan: bool = True  #  use parallel scan mode or sequential mode when training

            #
            self.TWO_OPERATION_RATIO = 3.

            # gene_key
            self.ImgEnc = "ImgEnc"
            self.AudEnc = "AudEnc"
            self.Inter = "Inter"
            self.Fusion = "Fusion"


    __C = CfgSearch()

    # inputs1 = torch.tensor([1.,0,0,0,0,0])
    # inputs2 = torch.tensor([10] * 6)
    # op = nn.Softmax(dim=-1)
    # print(op(inputs2 + inputs1))

    op = SelfAtt(__C,base=5).to('cuda:1')
    #
    for n, p in op.named_parameters():
        print(n)
    #
    inputs = (2 * torch.ones((2,3,5))).to('cuda:1')

    output = op(inputs)
    #
    print(output)