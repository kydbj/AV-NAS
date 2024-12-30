import torch
import numpy as np
import math, random, json
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from transformers import BertModel, BertConfig
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from model.pscan import pscan

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-12, dim=-1):
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

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x, y=None, x_mask=None, y_mask=None, rel_embed=None):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

''' 
==================
    cro_att
==================
'''
class CrossAtt(nn.Module):
    def __init__(self, __C, D, nrom=False, residual=False, base=64):
        super(CrossAtt, self).__init__()
        self.att = BertSelfAttention(__C, D, base=base)
        self.dense = nn.Linear(__C.HSIZE, __C.HSIZE)
        self.dropout = nn.Dropout(__C.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(__C.HSIZE, eps=__C.layer_norm_eps)

    def forward(self, x, y) -> torch.Tensor:
        hidden_states = self.att(x, y)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + x)
        return hidden_states

''' 
==================
    self_att
==================
'''
class BertSelfAttention(nn.Module):
    def __init__(self, __C, D, norm=False, residual=False, base=64):
        super(BertSelfAttention, self).__init__()

        self.HBASE = base
        self.HSIZE_INSIDE = D
        assert self.HSIZE_INSIDE % self.HBASE == 0
        self.HHEAD = int(self.HSIZE_INSIDE / self.HBASE)

        self.num_attention_heads = int(self.HSIZE_INSIDE / self.HBASE)
        self.attention_head_size = self.HBASE
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(D, self.all_head_size)
        self.key = nn.Linear(D, self.all_head_size)
        self.value = nn.Linear(D, self.all_head_size)

        self.dropout = nn.Dropout(__C.attention_probs_dropout_prob)


    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        x,
        y=None,
        attention_mask=None
    ):

        q = x
        if y is not None:
            k = y
            v = y
        else:
            k = x
            v = x

        mixed_query_layer = self.query(q)
        key_layer = self.transpose_for_scores(self.key(k))
        value_layer = self.transpose_for_scores(self.value(v))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer

''' 
==================
    att_output
==================
'''
class BertSelfOutput(nn.Module):
    def __init__(self, __C, D, norm=False, residual=False):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(D, D)
        self.dropout = nn.Dropout(__C.hidden_dropout_prob)
        # self.LayerNorm = nn.LayerNorm(__C.HSIZE, eps=__C.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


''' 
==================
    intermediate_output
==================
'''
class BertIntermediateOutput(nn.Module):
    def __init__(self, __C, D, norm=False, residual=False):
        super().__init__()
        self.dense1 = nn.Linear(D, __C.intermediate_size)
        self.intermediate_act_fn = nn.functional.gelu
        self.dense2 = nn.Linear(__C.intermediate_size, D)
        self.dropout = nn.Dropout(__C.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


''' 
==================
    卷积操作
==================
'''
class Dconv(nn.Module):
    def __init__(self, __C, D, k=None):
        super(Dconv, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=D, out_channels=D,
                                kernel_size=k, bias=__C.conv_bias,
                                groups=D,
                                padding=k - 1)
    def forward(self, x):

        _, L, _ = x.shape

        # x branch
        x = x.transpose(1, 2) # (B, ED, L)
        x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2) # (B, L, ED)

        return x


''' 
==================
    mlp_up
==================
'''
class MLP_UP(nn.Module):
    def __init__(self, __C, norm=False, residual=False):
        super(MLP_UP, self).__init__()
        self.in_proj = nn.Linear(__C.HSIZE, __C.d_inner,bias=__C.bias)

    def forward(self, x):
        x = self.in_proj(x) # (B, L, ED)
        return x

''' 
==================
    mlp_down
==================
'''
class MLP_DOWN(nn.Module):
    def __init__(self, __C, norm=False, residual=False):
        super(MLP_DOWN, self).__init__()
        self.in_proj = nn.Linear(__C.d_inner, __C.HSIZE,bias=__C.bias)

    def forward(self, x):
        x = self.in_proj(x) # (B, L, D)
        return x


''' 
==================
    self_ssm
==================
'''
class SelfSSM(nn.Module):
    def __init__(self, __C, D, norm=False, residual=False):
        super(SelfSSM, self).__init__()
        self.__C = __C
        self.norm = norm
        self.residual = residual

        self.ssm = SSM(__C, D)

        if norm:
            self.ln = LayerNorm(D)

    def forward(self,x):
        x_ssm = self.ssm(x,x)

        if self.norm:
            x_ssm = self.ln(x_ssm)
        return x_ssm

class SSM(nn.Module):
    def __init__(self, __C, D):
        super().__init__()
        self.__C = __C
        self.D = D

        #  projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(D, __C.dt_rank + 2 * __C.d_state, bias=False)
        #  projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(__C.dt_rank, D, bias=True)

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
            torch.rand(D) * (math.log(__C.dt_max) - math.log(__C.dt_min)) + math.log(__C.dt_min)
        ).clamp(min=__C.dt_init_floor)
        inv_dt = dt + torch.log(
            -torch.expm1(-dt))  #  inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        #  todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, __C.d_state + 1, dtype=torch.float32).repeat(D, 1)
        self.A_log = nn.Parameter(
            torch.log(A))  # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.D = nn.Parameter(torch.ones(D))

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

        h = torch.zeros(x.size(0), self.D, self.__C.d_state, device=deltaA.device)  #  (B, ED, N)
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
            h = torch.zeros(x.size(0), self.D, self.__C.d_state, device=deltaA.device)  #  (B, ED, N)

        h = deltaA * h + BX  #  (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  #  (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h.squeeze(1)


''' 
==================
    gated_att
==================
'''
class Gate_Att(nn.Module):
    def __init__(self, __C, norm=False, residual=False):
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


''' 
==================
    hashLayer
==================
'''
class HashLayer(nn.Module):

    def __init__(self, __C):
        super(HashLayer, self).__init__()
        self.__C = __C

        self.linear = nn.Linear(__C.HSIZE, __C.HASHCODE_SIZE)
        self.hashac = nn.Tanh()

    def forward(self, x):
        x = self.hashac(self.linear(x))
        return x


''' 
==================
    cro_ssm
==================
'''
class MambaCrossModel(nn.Module):
    def __init__(self, __C, norm=False, residual=False):
        super().__init__()

        self.__C = __C

        self.layers = ResidualCrossBlock(__C)
        # self.norm_f = RMSNorm(config.d_model)

    def forward(self, x1, x2):
        #  x : (B, L, D)

        #  y : (B, L, D)

        x =  self.layers(x1, x2)

        # x = self.norm_f(x)

        return x

class ResidualCrossBlock(nn.Module):
    def __init__(self, __C):
        super().__init__()
        self.__C = __C
        self.mixer = MambaCrossBlock(__C)
        self.norm1 = RMSNorm(__C.d_model)
        self.norm2 = RMSNorm(__C.d_model)

    def forward(self, x1, x2):
        #  x1, x2 : (B, L, D)

        #  output : (B, L, D)
        # output1 = self.mixer(self.norm1(x1), self.norm2(x2)) + x1 * self.Bias + x2 * (1 - self.Bias)
        output1 = self.mixer(self.norm1(x1), self.norm2(x2))

        return output1

    def step(self, x, cache):
        #  x : (B, D)
        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs: (B, ED, d_conv-1)

        #  output : (B, D)
        #  cache : (h, inputs)

        output, cache = self.mixer.step(self.norm1(x), cache)
        output = output + x
        return output, cache

class MambaCrossBlock(nn.Module):
    def __init__(self, __C):
        super().__init__()
        self.__C = __C

        #  projects block input from D to 2*ED (two branches)
        self.in_proj1 = nn.Linear(__C.d_model, 3 * __C.d_inner, bias=__C.bias)
        self.in_proj2 = nn.Linear(__C.d_model, __C.d_inner, bias=__C.bias)

        self.conv1d1 = nn.Conv1d(in_channels=__C.d_inner, out_channels=__C.d_inner,
                                 kernel_size=__C.d_conv, bias=__C.conv_bias,
                                 groups=__C.d_inner,
                                 padding=__C.d_conv - 1)

        self.conv1d2 = nn.Conv1d(in_channels=__C.d_inner, out_channels=__C.d_inner,
                                 kernel_size=__C.d_conv, bias=__C.conv_bias,
                                 groups=__C.d_inner,
                                 padding=__C.d_conv - 1)


        #  projects x to input-dependent Δ, B, C
        self.x2_proj = nn.Linear(__C.d_inner, __C.dt_rank + 2 * __C.d_state, bias=False)
        #  projects Δ from dt_rank to d_inner
        self.dt_proj2 = nn.Linear(__C.dt_rank, __C.d_inner, bias=True)

        #  dt initialization
        #  dt weights
        dt_init_std2 = __C.dt_rank ** -0.5 * __C.dt_scale
        if __C.dt_init == "constant":
            nn.init.constant_(self.dt_proj2.weight, dt_init_std2)
        elif __C.dt_init == "random":
            nn.init.uniform_(self.dt_proj2.weight, -dt_init_std2, dt_init_std2)
        else:
            raise NotImplementedError

        # dt bias   inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        dt2 = torch.exp(
            torch.rand(__C.d_inner) * (math.log(__C.dt_max) - math.log(__C.dt_min)) + math.log(__C.dt_min)
        ).clamp(min=__C.dt_init_floor)
        inv_dt2 = dt2 + torch.log(-torch.expm1(-dt2))
        with torch.no_grad():
            self.dt_proj2.bias.copy_(inv_dt2)
        # self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        #  todo : explain why removed
        # self.Bias1 =  nn.Parameter(torch.tensor(config.modal_bias-config.modal_bias+0.5))
        # self.Bias2 =  nn.Parameter(torch.tensor(config.modal_bias-config.modal_bias+0.5))

        # S4D real initialization
        A2 = torch.arange(1, __C.d_state + 1, dtype=torch.float32).repeat(__C.d_inner, 1)
        self.A2_log = nn.Parameter(torch.log(A2))
        self.A2b_log = nn.Parameter(torch.log(A2))

        #  projects block output from ED back to D
        self.out_proj = nn.Linear(__C.d_inner, __C.d_model, bias=__C.bias)

    def forward(self, x1, x2):
        #  x : (B, L, D)

        # y : (B, L, D)

        _B, L, _ = x1.shape

        x1 = self.in_proj1(x1)  # (B, L, 3*ED)
        x2 = self.in_proj2(x2)  # (B, L, 1*ED)
        x1, t1, z1 = x1.chunk(3, dim=-1)  #  (B, L, ED), (B, L, ED)

        _, _D1, _ = x1.shape
        _, _D2, _ = x2.shape
        #  x branch

        x1 = x1.transpose(1, 2)  #  (B, ED, L)
        x1 = self.conv1d1(x1)[:, :, :L]  #  depthwise convolution over time, with a short filter
        x1 = x1.transpose(1, 2)  #  (B, L, ED)

        x2 = x2.transpose(1, 2)  #  (B, ED, L)
        x2 = self.conv1d2(x2)[:, :, :L]  #  depthwise convolution over time, with a short filter
        x2 = x2.transpose(1, 2)  #  (B, L, ED)


        delta, B, C = torch.split(self.x2_proj(x2), [self.__C.dt_rank, self.__C.d_state, self.__C.d_state],
                                  dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj2(delta))  #  (B, L, ED)
        delta = delta.contiguous().float().transpose(1, 2)
        B = B.transpose(1, 2).float().view(_B, 1, self.__C.d_state, L)
        C = C.transpose(1, 2).float().view(_B, 1, self.__C.d_state, L)

        x1 = x1.transpose(1, 2)
        A2 = -torch.exp(self.A2_log.float())  # (ED, N)
        y1 = selective_scan_fn(x1, delta, A2, B, C, D=None, z=None, return_last_state=False, )
        # y1 = selective_scan_fn(x1, delta, A1, B, C, D=D1, z=None, return_last_state=False,)

        y1 = y1.transpose(1, 2)
        del A2
        A2b = -torch.exp(self.A2b_log.float())  # (ED, N)
        y1b = selective_scan_fn(torch.flip(x1, dims=[-1]), delta, A2b, B, C, D=None, z=None, return_last_state=False, )
        # y1b = selective_scan_fn(x1, delta, A1b, torch.flip(B,dims=[-1]), torch.flip(C,dims=[-1]), D=None, z=None, return_last_state=False,)
        # y1b = selective_scan_fn(torch.flip(x1,dims=[-1]), delta, A1b, B, C, D=torch.flip(D1,dims=[-1]), z=None, return_last_state=False,)
        # y1b = selective_scan_fn(torch.flip(x1,dims=[-1]), delta, A1b, torch.flip(B,dims=[-1]), torch.flip(C,dims=[-1]), D=None, z=None, return_last_state=False,)
        y1b = torch.flip(y1b, dims=[-1])

        # y1b = selective_scan_fn(torch.cat((x1[:,:,int(L/2):], x1[:,:,:int(L/2)]),dim=-1), delta, A1b, B, C, D=None, z=None, return_last_state=False,)

        y1b = y1b.transpose(1, 2)

        y1 = (y1 + y1b + t1) * F.silu(z1)
        # y1 = y1 + y1b + t1 * F.silu(z1) #3
        # y1 = (y1 + y1b + t1) * F.silu(z2) #4
        return self.out_proj(y1) #  (B, L, D)

    def ssm(self, x):
        #  x : (B, L, ED)

        #  y : (B, L, ED)
        _B, _L, _D = x.shape
        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()
        #  TODO remove .float()

        delta, B, C = torch.split(self.x_proj(x), [self.__C.dt_rank, self.__C.d_state, self.__C.d_state],
                                  dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, L, ED)
        delta = delta.contiguous().float().transpose(1, 2)
        B = B.transpose(1, 2).float().view(_B, 1, self.config.d_state, _L)
        C = C.transpose(1, 2).float().view(_B, 1, self.config.d_state, _L)
        x = x.transpose(1, 2)
        if self.config.pscan:
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
    The MambaCrossBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

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
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]  #  (B, ED)

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

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, dt_rank), (B, N), (B, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  #  (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)

        h = deltaA * h + BX  #  (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  #  (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        #  todo : pq h.squeeze(1) ??
        return y, h.squeeze(1)

#  taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

'''
MambaModel
'''
class MambaModel(nn.Module):
    def __init__(self, __C, norm=False, residual=False):
        super().__init__()

        self.__C = __C

        self.layers = nn.ModuleList([ResidualBlock(__C)])
        # self.norm_f = RMSNorm(config.d_model)

    def forward(self, x):
        #  x : (B, L, D)

        #  y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        return x

    def step(self, x, caches):
        #  x : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        #  y : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches



''' 
==================
    final_att
==================
'''
class EncoderModel(nn.Module):
    """ text processed by bert model encode and get cls vector for multi classification
    """

    def __init__(self, __C, norm=None, residual=None, model_config_path='/home/sunzhe/NAS_newArch/model', cls_path='/home/sunzhe/NAS_newArch/model/cls.pt'):
        super(EncoderModel, self).__init__()
        self.encoder = BertModel.from_pretrained(model_config_path)
        self.cls = torch.load(cls_path).float()
        # self.cls = torch.nn.Parameter(cls.float(), requires_grad=True) # 原本没有这行 这是可训练的

    # 采用预训练的cls
    def forward(self, x):
        '''
        加上一个cls的处理：预训练，随机初始化
        已经加上了
        '''
        cls = self.cls.to(x.device)
        batch = x.shape[0]
        expanded_cls = cls.expand(batch, -1)
        input = torch.cat((expanded_cls.unsqueeze(1), x), dim=1)
        output = self.encoder(inputs_embeds = input)

        return output.pooler_output


''' 
==================
    final_ssm
==================
'''
class FinalSSM(nn.Module):
    def __init__(self, __C, norm=False, residual=False):
        super().__init__()

        self.__C = __C

        self.layers = nn.ModuleList([ResidualBlock(__C)])
        # self.norm_f = RMSNorm(config.d_model)

    def forward(self, x):
        #  x : (B, L, D)

        #  y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        # x = self.norm_f(x)
        x = torch.mean(x, dim=1)

        return x

    def step(self, x, caches):
        #  x : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        #  y : (B, L, D)
        #  caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches

class ResidualBlock(nn.Module):
    def __init__(self, __C):
        super().__init__()
        self.__C = __C

        self.mixer = MambaBlock(__C)
        self.norm1 = RMSNorm(__C.d_model)

    def forward(self, x):
        #  x : (B, L, D)

        #  output : (B, L, D)
        output1 = self.mixer(self.norm1(x)) + x
        return output1

    def step(self, x, cache):
        #  x : (B, D)
        #  cache : (h, inputs)
        # h : (B, ED, N)
        #  inputs: (B, ED, d_conv-1)

        #  output : (B, D)
        #  cache : (h, inputs)

        output, cache = self.mixer.step(self.norm1(x), cache)
        output = output + x
        return output, cache

class MambaBlock(nn.Module):
    def __init__(self, __C):
        super().__init__()

        self.__C = __C

        #  projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(__C.d_model, 2 * __C.d_inner, bias=__C.bias)

        self.conv1d = nn.Conv1d(in_channels=__C.d_inner, out_channels=__C.d_inner,
                                kernel_size=__C.d_conv, bias=__C.conv_bias,
                                groups=__C.d_inner,
                                padding=__C.d_conv - 1)

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

        #  projects block output from ED back to D
        self.out_proj = nn.Linear(__C.d_inner, __C.d_model, bias=__C.bias)

    def forward(self, x):
        #  x : (B, L, D)

        # y : (B, L, D)

        _, L, _ = x.shape

        x = self.in_proj(x)  # (B, L, 2*ED)
        x, z = x.chunk(2, dim=-1)  #  (B, L, ED), (B, L, ED)

        #  x branch
        x = x.transpose(1, 2)  #  (B, ED, L)
        x = self.conv1d(x)[:, :, :L]  #  depthwise convolution over time, with a short filter
        x = x.transpose(1, 2)  #  (B, L, ED)

        x = F.silu(x)

        y = self.ssm(x)

        #  z branch
        x = F.silu(z)

        y = y * x
        y = self.out_proj(y)  #  (B, L, D)

        return y

    def ssm(self, x):
        #  x : (B, L, ED)

        #  y : (B, L, ED)
        _B, _L, _D = x.shape
        A = -torch.exp(self.A_log.float())  # (ED, N)
        D = self.D.float()
        #  TODO remove .float()

        delta, B, C = torch.split(self.x_proj(x), [self.__C.dt_rank, self.__C.d_state, self.__C.d_state],
                                  dim=-1)  #  (B, L, dt_rank), (B, L, N), (B, L, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, L, ED)
        delta = delta.contiguous().float().transpose(1, 2)
        B = B.transpose(1, 2).float().view(_B, 1, self.__C.d_state, _L)
        C = C.transpose(1, 2).float().view(_B, 1, self.__C.d_state, _L)
        x = x.transpose(1, 2)
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
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv - 1]  #  (B, ED)

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

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state],
                                  dim=-1)  #  (B, dt_rank), (B, N), (B, N)
        delta = F.softplus(self.dt_proj(delta))  #  (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  #  (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1)  #  (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1))  #  (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device)  #  (B, ED, N)

        h = deltaA * h + BX  #  (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2)  #  (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        #  todo : pq h.squeeze(1) ??
        return y, h.squeeze(1)




if __name__ == '__main__':
    class CfgSearch():
        def __init__(self):
            super(CfgSearch, self).__init__()

            self.DEVICE = 'cuda:0'  # 参数 1
            self.DATA_SET_CONFIG = './json/Anet.json'  # 参数 2
            self.LR = 1e-4
            self.HASHCODE_SIZE = 64
            self.MAX_ITER = 300
            self.WEIGHT_PATH = None

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
            self.BATCH_SIZE = 64
            self.EVAL_BATCH_SIZE = self.BATCH_SIZE

            # Network Params
            self.HSIZE = 768
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
            self.bias: bool = False
            self.modal_bias: float = 0.8
            self.pscan: bool = True  #  use parallel scan mode or sequential mode when training
            # Transformer Params
            self.attention_probs_dropout_prob = 0.1
            self.layer_norm_eps = 1e-12
            self.hidden_dropout_prob = 0.1
            self.intermediate_size = 3072

            # Optimizer Params
            self.NET_OPTIM = 'wadam'
            self.REDUCTION = 'sum'
            # self.REDUCTION = 'mean'
            # self.NET_OPTIM = 'sgd'

            if self.NET_OPTIM == 'sgd':
                self.NET_LR_BASE = 0.005
                self.NET_LR_MIN = 0.0005
                self.NET_MOMENTUM = 0.9
                # self.NET_WEIGHT_DECAY = 1e-4
                self.NET_WEIGHT_DECAY = 0
                self.NET_GRAD_CLIP = 5.  # GRAD_CLIP = -1: means not use grad_norm_clip 0.01
                self.MAX_EPOCH = 50

            else:
                self.NET_OPTIM_WARMUP = True
                self.NET_LR_BASE = 0.0004
                self.NET_WEIGHT_DECAY = 0
                self.NET_GRAD_CLIP = 1.  # GRAD_CLIP = -1: means not use grad_norm_clip
                self.NET_LR_DECAY_R = 0.2
                self.NET_LR_DECAY_LIST = []
                self.OPT_BETAS = (0.9, 0.98)
                self.OPT_EPS = 1e-9
                self.MAX_EPOCH = 100

            self.ALPHA_START = 20
            self.ALPHA_EVERY = 5

            self.ALPHA_LR_BASE = 0.1
            self.ALPHA_WEIGHT_DECAY = 0
            self.ALPHA_OPT_BETAS = (0., 0.999)

            # gene_key
            self.ImgEnc = "ImgEnc"
            self.AudEnc = "AudEnc"
            self.Inter = "Inter"
            self.Fusion = "Fusion"
            self.Final = "Fianl"

            # operators
            self.TWO_OPERATION_RATIO = 1


    __C = CfgSearch()

    # inputs1 = torch.tensor([1.,0,0,0,0,0])
    # inputs2 = torch.tensor([10] * 6)
    # op = nn.Softmax(dim=-1)
    # print(op(inputs2 + inputs1))
    conv1d = nn.Conv1d(in_channels=768, out_channels=768,
                            kernel_size=4, bias=False,
                            groups=768,
                            padding=4 - 1)

    for n, p in conv1d.named_parameters():
        print(p.shape)
    # op = EncoderModel(__C,model_config_path='../model',cls_path='../model/cls.pt')

    # op = MLP_STD(__C)
    # op = StdConv(__C, k=3)
    # op = MambaCrossModel(__C).to('cuda:1')
    # op = FinalSSM(__C).to('cuda:1')
    #
    # for n, p in op.named_parameters():
    #     print(n)
    #
    # inputs1 = torch.rand((2,3,5)).to('cuda:1')
    # # inputs2 = torch.rand((2,3,5)).to('cuda:1')
    # out = op(inputs1)
    # print(out.shape)

