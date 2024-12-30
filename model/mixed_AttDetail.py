import numpy as np
import math, random, json
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from utils.ops_adapter_new import OpsAdapter

OPS_ADAPTER = OpsAdapter()

class MixedOp(nn.Module):
    '''
    ==================
        name = MonadicOperators_high MonadicOperators_low BinaryOperators ModelingOperators_high ModelingOperators_low
        InterOperators FusionOperators FinalOperators
    ==================
    '''
    def __init__(self, __C, op_name:str=None, op_list:list[str]=None, retrainWithAlpha=False):
        super().__init__()
        self.__C = __C
        self.alpha_prob = None
        if op_name in OPS_ADAPTER.Used_OPS: # search
            self.Used_OPS = OPS_ADAPTER.Used_OPS[op_name]
            self.n_choices = len(self.Used_OPS)
            # self.alpha_prob = nn.Parameter(torch.zeros(self.n_choices))
            self.alpha_prob = nn.Parameter(torch.randn(self.n_choices))

        else: # retrain
            self.Used_OPS = op_list
            if retrainWithAlpha:
                self.n_choices = len(self.Used_OPS)
                if self.n_choices > 1:
                    self.alpha_prob = nn.Parameter(torch.zeros(self.n_choices))

        self.candidate_ops = nn.ModuleDict()
        for op_name in self.Used_OPS:
            op = OPS_ADAPTER.OPS[op_name](__C, norm=__C.OPS_NORM, residual=__C.OPS_RESIDUAL)
            self.candidate_ops[op_name] = op

    def forward(self, x, y=None):
        if self.alpha_prob is not None:  # search
            if y is not None:  # tow inputs
                return sum(w * op(x, y) for w, op in zip(self.probs_over_ops, self.candidate_ops.values()))
            else:   # one input
                return sum(w * op(x) for w, op in zip(self.probs_over_ops, self.candidate_ops.values()))
        else:  # retrain
            if y is not None:  # tow inputs
                return sum(op(x, y) for op in self.candidate_ops.values())
            else:   # one input
                return sum(op(x) for op in self.candidate_ops.values())

    @property
    def probs_over_ops(self):
        probs = F.softmax(self.alpha_prob, dim=0)  # softmax to probability
        return probs



if __name__  == '__main__':
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
            self.expand_factor: int = 2  #  E in paper/comments
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

    """
            name = 
            MonadicOperators_high MonadicOperators_low 
            BinaryOperators 
            ModelingOperators_high ModelingOperators_low
            InterOperators FusionOperators FinalOperators
    """

    __C = CfgSearch()
    model = MixedOp(__C, "FinalOperators").to("cuda:0")
    # inputs1 = torch.rand((2,3,1536)).to('cuda:0')
    # inputs2 = torch.rand((2,3,1536)).to('cuda:0')
    inputs1 = torch.rand((2,3,768)).to('cuda:0')
    # inputs2 = torch.rand((2,3,768)).to('cuda:0')
    out = model(inputs1)
    print(out.shape)

    # for (n,p) in model.named_parameters():
    #     print(n)


