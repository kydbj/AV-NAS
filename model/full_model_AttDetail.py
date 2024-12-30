from utils.ops_adapter_new import OpsAdapter

# from model.mixed import MixedOp
from model.mixed_AttDetail import MixedOp
import numpy as np
import math, random, json
import torch
import torch.nn as nn
from model.modules import HashLayer
import torch.nn.functional as F
from utils.log import log_training_data, combine_genotype_probs

OPS_ADAPTER = OpsAdapter()

# 用于 retrain
'''     
ModelingOperators_high ModelingOperators_low

MonadicOperators_high MonadicOperators_low 

BinaryOperators InterOperators FusionOperators FinalOperators
'''

class Encoder_Full(nn.Module):
    def __init__(self,__C, type):
        super(Encoder_Full, self).__init__()
        self.__C = __C
        self.up1 = nn.Linear(__C.HSIZE, __C.d_inner,bias=__C.bias)
        self.up2 = nn.Linear(__C.HSIZE, __C.d_inner,bias=__C.bias)
        self.down = nn.Linear(__C.d_inner, __C.HSIZE,bias=__C.bias)

        gene = __C.GENOTYPE[type]
        # 2 ModelingOperators_high 1 ModelingOperators_low
        # 2 MonadicOperators_high 2 MonadicOperators_low
        # 2 BinaryOperators
        self.ModelingOperators_high = nn.ModuleList()
        self.ModelingOperators_low = nn.ModuleList()
        self.MonadicOperators_high = nn.ModuleList()
        self.MonadicOperators_low = nn.ModuleList()
        self.BinaryOperators = nn.ModuleList()

        ops = gene["ModelingOperators_high"]
        for op_list in ops:
            op = MixedOp(__C, op_list=op_list, retrainWithAlpha=self.__C.retrainWithAlpha)
            self.ModelingOperators_high.append(op)

        ops = gene["ModelingOperators_low"]
        for op_list in ops:
            op = MixedOp(__C, op_list=op_list, retrainWithAlpha=self.__C.retrainWithAlpha)
            self.ModelingOperators_low.append(op)

        ops = gene["MonadicOperators_high"]
        for op_list in ops:
            op = MixedOp(__C, op_list=op_list, retrainWithAlpha=self.__C.retrainWithAlpha)
            self.MonadicOperators_high.append(op)

        ops = gene["MonadicOperators_low"]
        for op_list in ops:
            op = MixedOp(__C, op_list=op_list, retrainWithAlpha=self.__C.retrainWithAlpha)
            self.MonadicOperators_low.append(op)

        ops = gene["BinaryOperators"]
        for op_list in ops:
            op = MixedOp(__C, op_list=op_list, retrainWithAlpha=self.__C.retrainWithAlpha)
            self.BinaryOperators.append(op)

    def forward(self, x):
        xup1 = self.up1(x)
        xup2 = self.up2(x)

        model_h1 = self.ModelingOperators_high[1](self.MonadicOperators_high[0](self.ModelingOperators_high[0](xup1)))
        mona_h1 = self.MonadicOperators_high[1](xup2)
        xdown = self.down(self.BinaryOperators[0](model_h1, mona_h1))
        mona_l0 = self.MonadicOperators_low[0](xdown)
        model_l = self.ModelingOperators_low[0](mona_l0)
        out = self.MonadicOperators_low[1](self.BinaryOperators[1](model_l, mona_l0))

        return out

class Interact_Full(nn.Module):
    def __init__(self,__C, type):
        super(Interact_Full, self).__init__()
        self.__C = __C
        
        gene = __C.GENOTYPE[type]
        self.InterOperators = nn.ModuleList()

        ops = gene["InterOperators"]
        for op_list in ops:
            op = MixedOp(__C, op_list=op_list,retrainWithAlpha=self.__C.retrainWithAlpha)
            self.InterOperators.append(op)

    def forward(self, x, y):
        Xout = self.InterOperators[0](x,y)
        Yout = self.InterOperators[1](y,x)

        return Xout, Yout

class Fusion_Full(nn.Module):
    def __init__(self,__C, type):
        super(Fusion_Full, self).__init__()
        self.__C = __C

        gene = __C.GENOTYPE[type]
        self.FusionOperators = nn.ModuleList()

        ops = gene["FusionOperators"]
        for op_list in ops:
            op = MixedOp(__C, op_list=op_list, retrainWithAlpha=self.__C.retrainWithAlpha)
            self.FusionOperators.append(op)

    def forward(self, x, y):
        F = self.FusionOperators[0](x, y)

        return F

class Final_Full(nn.Module):
    def __init__(self, __C, type):
        super(Final_Full, self).__init__()
        self.__C = __C

        gene = __C.GENOTYPE[type]
        self.FinalOperators = nn.ModuleList()

        ops = gene["FinalOperators"]
        for op_list in ops:
            op = MixedOp(__C, op_list=op_list, retrainWithAlpha=self.__C.retrainWithAlpha)
            self.FinalOperators.append(op)

    def forward(self, x):
        F = self.FinalOperators[0](x)
        return F

class Beckbone_Full(nn.Module):
    def __init__(self,__C):
        super(Beckbone_Full, self).__init__()
        self.__C = __C

        self.Img_encoder = Encoder_Full(self.__C, type=__C.ImgEnc)
        self.Aud_encoder = Encoder_Full(self.__C, type=__C.AudEnc)

        self.Interaction = Interact_Full(self.__C, type=__C.Inter)
        self.Fusion = Fusion_Full(self.__C,type=__C.Fusion)
        self.Final = Final_Full(self.__C, type=__C.Final)


    def forward(self, img, aud):
        A = self.Img_encoder(img)
        B = self.Aud_encoder(aud)

        C,D = self.Interaction(A,B)
        E = self.Fusion(C,D)
        F = self.Final(E)

        return F

class Net_Full(nn.Module):
    def __init__(self, __C):
        super(Net_Full, self).__init__()
        self.__C = __C

        # log_training_data(OPS_ADAPTER.Used_OPS, self.__C.LOG_PATH)

        self.net = Beckbone_Full(__C)
        self.hash = HashLayer(__C)
        # setup net_weights list
        self._net_weights = []
        for n, p in self.named_parameters():
            if 'alpha_prob' not in n:
                self._net_weights.append((n, p))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha_prob' in n:
                self._alphas.append((n, p))

    def forward(self, x, y):
        A = self.net(x, y)
        # A_pool = torch.mean(A, dim=1)
        out = self.hash(A)
        return out

    def alpha_prob_parameters(self):
        for n, p in self._alphas:
            yield p

    def named_alpha_prob_parameters(self):
        for n, p in self._alphas:
            yield n,p

    def net_parameters(self):
        for n, p in self._net_weights:
            yield p

    def named_net_parameters(self):
        for n, p in self._net_weights:
            yield n, p



    def genotype(self):

        operators = ["ModelingOperators_high", "ModelingOperators_low", "MonadicOperators_high", "MonadicOperators_low",
                     "BinaryOperators", "InterOperators", "FusionOperators", "FinalOperators"]

        alpha_ImgEnc = []  # cell for every mixop
        ImgEnc_type = []   # optype for every mixop
        alpha_AudEnc = []
        AudEnc_type = []
        alpha_Interaction = []
        Interaction_type = []
        alpha_Fusion = []
        Fusion_type = []
        alpha_Final = []
        Final_type = []

        for n, p in self.named_alpha_prob_parameters():

            type = None
            for op in operators:
                if op in n:
                    type = op

            if 'Img_encoder' in n:
                alpha_ImgEnc.append(p)
                ImgEnc_type.append(type)
            if 'Aud_encoder' in n:
                alpha_AudEnc.append(p)
                AudEnc_type.append(type)
            if 'Interaction' in n:
                alpha_Interaction.append(p)
                Interaction_type.append(type)
            if 'Fusion' in n:
                alpha_Fusion.append(p)
                Fusion_type.append(type)
            if 'Final' in n:
                alpha_Final.append(p)
                Final_type.append(type)

        gene_ImgEnc1, gene_ImgEnc_prob1, gene_ImgEnc2, gene_ImgEnc_prob2 = self.parse(alpha_ImgEnc, ImgEnc_type)
        gene_AudEnc1, gene_AudEnc_prob1, gene_AudEnc2, gene_AudEnc_prob2 = self.parse(alpha_AudEnc, AudEnc_type)
        gene_Interaction1, gene_Interaction_prob1, gene_Interaction2, gene_Interaction_prob2 = self.parse(alpha_Interaction, Interaction_type)
        gene_Fusion1, gene_Fusion_prob1, gene_Fusion2, gene_Fusion_prob2 = self.parse(alpha_Fusion, Fusion_type)
        gene_Final1, gene_Final_prob1, gene_Final2, gene_Final_prob2 = self.parse(alpha_Final, Final_type)

        #
        arch1 =  {
            self.__C.ImgEnc: gene_ImgEnc1,
            self.__C.AudEnc: gene_AudEnc1,
            self.__C.Inter: gene_Interaction1,
            self.__C.Fusion: gene_Fusion1,
            self.__C.Final: gene_Final1
        }

        arch_probs1 =  {
            self.__C.ImgEnc: gene_ImgEnc_prob1,
            self.__C.AudEnc: gene_AudEnc_prob1,
            self.__C.Inter: gene_Interaction_prob1,
            self.__C.Fusion: gene_Fusion_prob1,
            self.__C.Final: gene_Final_prob1
        }

        arch2 =  {
            self.__C.ImgEnc: gene_ImgEnc2,
            self.__C.AudEnc: gene_AudEnc2,
            self.__C.Inter: gene_Interaction2,
            self.__C.Fusion: gene_Fusion2,
            self.__C.Final: gene_Final2
        }

        arch_probs2 =  {
            self.__C.ImgEnc: gene_ImgEnc_prob2,
            self.__C.AudEnc: gene_AudEnc_prob2,
            self.__C.Inter: gene_Interaction_prob2,
            self.__C.Fusion: gene_Fusion_prob2,
            self.__C.Final: gene_Final_prob2
        }

        return arch1, arch_probs1, arch2, arch_probs2

    def parse(self, alpha_param_list, type_list):


        def parse_one(alpha_param_list, type_list):
            gene_one = {}
            gene_prob_one = {}

            with torch.no_grad():
                for ix, edges in enumerate(alpha_param_list):
                    type = type_list[ix]
                    if type not in gene_one:
                        gene_one[type] = []
                        gene_prob_one[type] = []
                    # edges: Tensor(n_edges, n_ops)
                    soft_edges = F.softmax(edges, dim=-1)
                    edge_max, op_indices = torch.topk(soft_edges, 1)  # choose top-1 op in every edge
                    op_name = [OPS_ADAPTER.Used_OPS[type][op_indices[0]]]
                    gene_one[type].append(op_name)
                    gene_prob_one[type].append([round(edge_max.item(), 4)])

            return gene_one, gene_prob_one

        def parse_two(alpha_param_list, type_list):
            gene_two = {}
            gene_prob_two = {}

            with torch.no_grad():
                for ix, edges in enumerate(alpha_param_list):
                    type = type_list[ix]
                    if type not in gene_two:
                        gene_two[type] = []
                        gene_prob_two[type] = []
                    # edges: Tensor(n_edges, n_ops)
                    soft_edges = F.softmax(edges, dim=-1)

                    if len(soft_edges) > 2 and type not in ["MonadicOperators_high", "MonadicOperators_low", "BinaryOperators"]:
                        edge_max, op_indices = torch.topk(soft_edges, 2)
                        if edge_max[0] / edge_max[1] <= self.__C.TWO_OPERATION_RATIO:
                            # 如果有两个， 还需要过滤掉None
                            valid_ops = []
                            valid_probs = []
                            for i in range(len(edge_max)):
                                op_name = OPS_ADAPTER.Used_OPS[type][op_indices[i]]
                                if op_name != "none":
                                    valid_ops.append(op_name)
                                    valid_probs.append(round(edge_max[i].item(), 4))
                            if valid_ops:
                                gene_two[type].append(valid_ops)
                                gene_prob_two[type].append(valid_probs)

                        else:
                            op_name = [OPS_ADAPTER.Used_OPS[type][op_indices[0]]]
                            gene_two[type].append(op_name)
                            gene_prob_two[type].append([round(edge_max[0].item(), 4)])
                    else: # 只有一个操作，或者是"BinaryOperators","MonadicOperators"这两类操作
                        edge_max, op_indices = torch.topk(soft_edges, 1)  # choose top-1 op in every edge
                        op_name = [OPS_ADAPTER.Used_OPS[type][op_indices[0]]]
                        gene_two[type].append(op_name)
                        gene_prob_two[type].append([round(edge_max.item(), 4)])

            return  gene_two, gene_prob_two

        A,B = parse_one(alpha_param_list, type_list)
        C,D = parse_two(alpha_param_list, type_list)

        return A, B, C, D

    def genotype_weights(self):

        operators = ["ModelingOperators_high", "ModelingOperators_low", "MonadicOperators_high", "MonadicOperators_low",
                     "BinaryOperators", "InterOperators", "FusionOperators", "FinalOperators"]
        alpha_ImgEnc = []  # cell for every mixop
        ImgEnc_type = []   # optype for every mixop
        alpha_AudEnc = []
        AudEnc_type = []
        alpha_Interaction = []
        Interaction_type = []
        alpha_Fusion = []
        Fusion_type = []
        alpha_Final = []
        Final_type = []

        for n, p in self.named_alpha_prob_parameters():

            type = None
            for op in operators:
                if op in n:
                    type = op

            if 'Img_encoder' in n:
                alpha_ImgEnc.append(p)
                ImgEnc_type.append(type)
            if 'Aud_encoder' in n:
                alpha_AudEnc.append(p)
                AudEnc_type.append(type)
            if 'Interaction' in n:
                alpha_Interaction.append(p)
                Interaction_type.append(type)
            if 'Fusion' in n:
                alpha_Fusion.append(p)
                Fusion_type.append(type)
            if 'Final' in n:
                alpha_Final.append(p)
                Final_type.append(type)

        gene_ImgEnc = self.parse_weights(alpha_ImgEnc, ImgEnc_type)
        gene_AudEnc = self.parse_weights(alpha_AudEnc, AudEnc_type)
        gene_Interaction = self.parse_weights(alpha_Interaction, Interaction_type)
        gene_Fusion = self.parse_weights(alpha_Fusion, Fusion_type)
        gene_Final = self.parse_weights(alpha_Final, Final_type)
        #
        arch_probs = {
            self.__C.ImgEnc: gene_ImgEnc,
            self.__C.AudEnc: gene_AudEnc,
            self.__C.Inter: gene_Interaction,
            self.__C.Fusion: gene_Fusion,
            self.__C.Final: gene_Final,
        }

        return arch_probs


    def parse_weights(self, alpha_param_list, type_list):
        with torch.no_grad():
            gene = {}
            for ix, edges in enumerate(alpha_param_list):
                type = type_list[ix]
                if type not in gene:
                    gene[type] = []
                # weight = edges.data.cpu().numpy().tolist()
                weight = F.softmax(edges, dim=-1).data.cpu().numpy().tolist()
                weight_list = [round(val, 4) for val in weight]
                gene[type].append(weight_list)
            return gene


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


    __C = CfgSearch()
    op = Fusion_Full(__C).to('cuda:1')
    inputs1 = torch.rand((2,3,768)).to('cuda:1')
    inputs2 = torch.rand((2,3,768)).to('cuda:1')
    out = op(inputs1, inputs2)
    print(out.shape)

    # __C = CfgSearch()
    # model = Net_Search(__C)
    # for n,p in model.named_parameters():
    #     print(n)
    #
    # A,B,C,D = model.genotype()
    #
    # print(A)
    # print(B)
    # print(C)
    # print(D)
