import torch.nn as nn
from model.modules_new1 import *


class OpsAdapter:
    def __init__(self):


        self.Used_OPS = {
            'MonadicOperators_high': [
                'skip_connect',
                'silu',
            ],

            'MonadicOperators_low': [
                'skip_connect',
                'ln_norm_l',
            ],

            'BinaryOperators': [
                'add',
                'hadamard'
            ],

            'ModelingOperators_high': [ # d_inner
                'skip_connect',
                'self_att_64_h',
                'self_ssm_h',
                'Dconv_3_h',
                'Dconv_4_h',
                'Dconv_5_h',
                'att_output_h', # 恒等变换
                'intermediate_output_h', # FFN 1536 * 4
            ],

            'ModelingOperators_low': [ # hsize
                'skip_connect',
                'self_att_64_l',
                'self_ssm_l',
                'Dconv_3_l',
                'Dconv_4_l',
                'Dconv_5_l',
                'att_output_l',  # 恒等变换
                'intermediate_output_l', # FFN 768 * 4
            ],

            'InterOperators': [
                # 'none',
                'cro_att_64',
                'cro_ssm'
            ],

            'FusionOperators': [
                'add',
                # 'self_att_64',
                'gated_att'
            ],

            'FinalOperators': [
                'final_att_64',
                'final_ssm'
            ]
        }

        self.OPS = {
            # 'none': lambda __C, norm, residual: Zero(),
            # 'relu': lambda __C, norm, residual: nn.ReLU(),
            # 'gelu': lambda __C, norm, residual: GELU(),
            # 'leakyrelu': lambda __C, norm, residual: nn.LeakyReLU(),

            # MonadicOperators_high
            'silu': lambda __C, norm, residual: nn.SiLU(),
            'skip_connect': lambda __C, norm, residual: Identity(__C, norm),
            # MonadicOperators_low
            'ln_norm_l': lambda __C, norm, residual: nn.LayerNorm(__C.HSIZE,eps=__C.layer_norm_eps),
            # BinaryOperators
            'add': lambda __C, norm, residual: Add(__C, norm),
            'hadamard': lambda __C, norm, residual: Hadamard(__C, norm),


            'self_att_256_h': lambda __C, norm, residual: BertSelfAttention(__C, __C.d_inner, norm, residual, base=256),
            'self_att_128_h': lambda __C, norm, residual: BertSelfAttention(__C, __C.d_inner, norm, residual, base=128),
            'self_att_64_h': lambda __C, norm, residual: BertSelfAttention(__C, __C.d_inner, norm, residual, base=64),
            'self_att_32_h': lambda __C, norm, residual: BertSelfAttention(__C, __C.d_inner, residual, base=32),
            'self_att_16_h': lambda __C, norm, residual: BertSelfAttention(__C, __C.d_inner, residual, base=16),

            'self_att_256_l': lambda __C, norm, residual: BertSelfAttention(__C, __C.HSIZE, residual, base=256),
            'self_att_128_l': lambda __C, norm, residual: BertSelfAttention(__C, __C.HSIZE, norm, residual, base=128),
            'self_att_64_l': lambda __C, norm, residual: BertSelfAttention(__C, __C.HSIZE, residual, base=64),
            'self_att_32_l': lambda __C, norm, residual: BertSelfAttention(__C, __C.HSIZE, residual, base=32),
            'self_att_16_l': lambda __C, norm, residual: BertSelfAttention(__C, __C.HSIZE, residual, base=16),

            'self_ssm_h': lambda __C, norm, residual: SelfSSM(__C, __C.d_inner, norm, residual),
            'self_ssm_l': lambda __C, norm, residual: SelfSSM(__C, __C.HSIZE, norm, residual),

            'Dconv_3_h': lambda __C, norm, residual: Dconv(__C, __C.d_inner, 3),
            'Dconv_4_h': lambda __C, norm, residual: Dconv(__C, __C.d_inner, 4),
            'Dconv_5_h': lambda __C, norm, residual: Dconv(__C, __C.d_inner, 5),

            'Dconv_3_l': lambda __C, norm, residual: Dconv(__C, __C.HSIZE, 3),
            'Dconv_4_l': lambda __C, norm, residual: Dconv(__C, __C.HSIZE, 4),
            'Dconv_5_l': lambda __C, norm, residual: Dconv(__C, __C.HSIZE, 5),


            'cro_att_256': lambda __C, norm, residual: CrossAtt(__C, __C.HSIZE, norm, residual, base=256),
            'cro_att_128': lambda __C, norm, residual: CrossAtt(__C, __C.HSIZE, norm, residual, base=128),
            'cro_att_64': lambda __C, norm, residual: CrossAtt(__C, __C.HSIZE, norm, residual, base=64),
            'cro_att_32': lambda __C, norm, residual: CrossAtt(__C, __C.HSIZE, norm, residual, base=32),
            'cro_att_16': lambda __C, norm, residual: CrossAtt(__C, __C.HSIZE, norm, residual, base=16),
            'cro_ssm': lambda __C, norm, residual: MambaCrossModel(__C, norm, residual),

            'final_att_64': lambda __C, norm, residual: EncoderModel(__C, norm, residual),
            'final_ssm': lambda __C, norm, residual: FinalSSM(__C, norm, residual),

            'gated_att': lambda __C, norm, residual: Gate_Att(__C, norm, residual),

            'att_output_h': lambda __C, norm, residual: BertSelfOutput(__C, __C.d_inner, norm, residual), #
            'att_output_l': lambda __C, norm, residual: BertSelfOutput(__C, __C.HSIZE, norm, residual), #

            'intermediate_output_h': lambda __C, norm, residual: BertIntermediateOutput(__C, __C.d_inner, norm, residual),
            'intermediate_output_l': lambda __C, norm, residual: BertIntermediateOutput(__C, __C.HSIZE, norm, residual),
        }