import json
from functools import reduce
from operator import mul
import torch.nn.functional as F
import torch

def probs_one(file='/mnt/add_disk/NAS/ActNet/Log_25/prob.json', j=None):


    # 从文件中读取 JSON 数据
    if file:
        with open(file, 'r') as file:
            data = json.load(file)

    if j:
        data = j


    def get_all_probabilities(epoch_data):
        all_probs = []
        for category in epoch_data.values():
            for operator_probs in category.values():
                for prob in operator_probs:
                    all_probs.append(prob[0])
        return all_probs


    max_epoch = None
    max_product = 0

    # 计算每个 epoch 中所有概率的乘积
    for epoch, epoch_data in data.items():
        probabilities = get_all_probabilities(epoch_data)
        product = reduce(mul, probabilities, 1)

        if product > max_product:
            max_product = product
            max_epoch = epoch

    print(f"The epoch with the maximum product is {max_epoch} with a product of {max_product}")

def probs_two(file='/mnt/add_disk/NAS/ActNet/Log_26/probs.json', OPS_ADAPTER=None):

    def genotype(epoch_data):


        # operators = ["ModelingOperators_high", "ModelingOperators_low", "MonadicOperators_high", "MonadicOperators_low",
        #              "BinaryOperators", "InterOperators", "FusionOperators", "FinalOperators"]

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


        for n, p in epoch_data.items():


            for op, probs in p.items():

                if 'ImgEnc' in n:
                    alpha_ImgEnc.extend(probs)
                    ImgEnc_type.extend([op] * len(probs))
                if 'AudEnc' in n:
                    alpha_AudEnc.extend(probs)
                    AudEnc_type.extend([op] * len(probs))
                if 'Inter' in n:
                    alpha_Interaction.extend(probs)
                    Interaction_type.extend([op] * len(probs))
                if 'Fusion' in n:
                    alpha_Fusion.extend(probs)
                    Fusion_type.extend([op] * len(probs))
                if 'Final' in n:
                    alpha_Final.extend(probs)
                    Final_type.extend([op] * len(probs))

        gene_ImgEnc2, gene_ImgEnc_prob2, gene_ImgEnc_sum = parse(alpha_ImgEnc, ImgEnc_type)
        gene_AudEnc2, gene_AudEnc_prob2, gene_AudEnc_sum = parse(alpha_AudEnc, AudEnc_type)
        gene_Interaction2, gene_Interaction_prob2, gene_Interaction_sum = parse(alpha_Interaction, Interaction_type)
        gene_Fusion2, gene_Fusion_prob2, gene_Fusion_sum = parse(alpha_Fusion, Fusion_type)
        gene_Final2, gene_Final_prob2, gene_Final_sum = parse(alpha_Final, Final_type)


        arch2 =  {
            "ImgEnc": gene_ImgEnc2,
            "AudEnc": gene_AudEnc2,
            "Inter": gene_Interaction2,
            "Fusion": gene_Fusion2,
            "Final": gene_Final2
        }

        arch_probs2 =  {
            "ImgEnc": gene_ImgEnc_prob2,
            "AudEnc": gene_AudEnc_prob2,
            "Inter": gene_Interaction_prob2,
            "Fusion": gene_Fusion_prob2,
            "Final": gene_Final_prob2
        }

        arch_probs2_sum =  {
            "ImgEnc": gene_ImgEnc_sum,
            "AudEnc": gene_AudEnc_sum,
            "Inter": gene_Interaction_sum,
            "Fusion": gene_Fusion_sum,
            "Final": gene_Final_sum
        }


        return arch2, arch_probs2, arch_probs2_sum

    def parse(alpha_param_list, type_list):



        def parse_two(alpha_param_list, type_list):
            gene_two = {}
            gene_prob_two = {}
            gene_sum_probs = {}

            for ix, edges in enumerate(alpha_param_list):
                type = type_list[ix]
                if type not in gene_two:
                    gene_two[type] = []
                    gene_prob_two[type] = []
                    gene_sum_probs[type] = []
                # edges: Tensor(n_edges, n_ops)
                soft_edges = F.softmax(torch.tensor(edges), dim=-1)

                if len(soft_edges) > 2 and type not in ["MonadicOperators_high", "MonadicOperators_low", "BinaryOperators"]:
                    edge_max, op_indices = torch.topk(soft_edges, 2)
                    if edge_max[0] / edge_max[1] <= 3:
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
                            gene_sum_probs[type].append([round(sum(valid_probs),4)])

                    else:
                        op_name = [OPS_ADAPTER.Used_OPS[type][op_indices[0]]]
                        gene_two[type].append(op_name)
                        gene_prob_two[type].append([round(edge_max[0].item(), 4)])
                        gene_sum_probs[type].append([round(edge_max[0].item(), 4)])

                else: # 只有一个操作，或者是"BinaryOperators","MonadicOperators"这两类操作
                    edge_max, op_indices = torch.topk(soft_edges, 1)  # choose top-1 op in every edge
                    op_name = [OPS_ADAPTER.Used_OPS[type][op_indices[0]]]
                    gene_two[type].append(op_name)
                    gene_prob_two[type].append([round(edge_max.item(), 4)])
                    gene_sum_probs[type].append([round(edge_max.item(), 4)])

            return  gene_two, gene_prob_two, gene_sum_probs

        C,D,E = parse_two(alpha_param_list, type_list)

        return C, D, E

    operators = ["ModelingOperators_high", "ModelingOperators_low", "MonadicOperators_high", "MonadicOperators_low",
                 "BinaryOperators", "InterOperators", "FusionOperators", "FinalOperators"]

    # 从文件中读取 JSON 数据
    with open(file, 'r') as file:
        data = json.load(file)

    sumprobs = []
    sumprobs_json = {}
    probabilities = []
    archs = []

    for epoch, epoch_data in data.items():
        infos = genotype(epoch_data)
        archs.append(infos[0])
        probabilities.append(infos[1])
        sumprobs.append(infos[2])
        sumprobs_json[epoch] = infos[2]

    probs_one(j=sumprobs_json)

    print(archs[298])
    print(archs[284])


from ops_adapter_new import OpsAdapter

OpsAdapter = OpsAdapter()
probs_one()
probs_two(OPS_ADAPTER=OpsAdapter)
