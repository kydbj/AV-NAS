# @file name  : test2.py
# @brief      :
# @author     : liupc
# @date       : 2021/8/2

# 需要标签


import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

# 计算汉明距离。有几位不同，距离就为几。
def CalcHammingDist_numpy(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - B1 @ B2.t())
    return distH

def chunked_hamming_distance(B1, B2, chunk_size=1000):
    q = B2.shape[1]
    distH = torch.zeros(B1.shape[0], B2.shape[0])

    epoch = B1.shape[0] // chunk_size
    rem = B1.shape[0] % chunk_size

    print(epoch, rem)
    for i in range(epoch):
        chunk_B1 = B1[i*chunk_size:i*chunk_size + chunk_size]
        chunk_distH = 0.5 * (q - chunk_B1 @ B2.t())
        distH[i*chunk_size:i*chunk_size + chunk_size] = chunk_distH

    if rem:
        chunk_B1 = B1[epoch*chunk_size:]
        chunk_distH = 0.5 * (q - chunk_B1 @ B2.t())
        distH[epoch*chunk_size:] = chunk_distH

    return distH

# 计算mAP@k的值。
def CalcTopMap(qB, rB, queryL, retrievalL, topk):
    # qB queryBinary, query数据集，都转成了哈希码
    # rB retrievalBinary，gallery数据集，被查询的数据集，都转成了哈希码。
    # queryL queryLabel，query数据集的标签
    # retrievalL retrievalLabel，gallery数据集的标签

    num_query = queryL.shape[0]  # 共有多少个查询
    topkmap = 0
    for iter in range(num_query):  # 对每一个查询，求其AP@k的值
        # gnd:一个长度等于gallery数据集的0/1向量，1表示为正样本(即至少包含query的一个标签)，0表示负样本（即，不包含query的标签）。
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)  # 计算第iter张query图片与gallery数据集的汉明距离
        ind = np.argsort(hamm)  # 对汉明距离进行排序，返回下标，表示最相似的图片列表
        gnd = gnd[ind]  # 最相似的图片是否为正样本的列表

        tgnd = gnd[0:topk]  # 只看前topk个结果
        tsum = np.sum(tgnd).astype(int)  # 前topk个中有多少个预测对了
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap += topkmap_
    topkmap = topkmap / num_query
    return topkmap


# 计算mAP的值
def CalcMap(qB, rB, queryL, retrievalL):
    # qB queryBinary, query数据集，都转成了哈希码
    # rB retrievalBinary，gallery数据集，都转成了哈希码
    # queryL queryLabel，query数据集的标签
    # retrievalL retrievalLabel，gallery数据集的标签

    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd).astype(int)
        if tsum == 0:
            continue
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query

    return map

def CalcTopAcc(qB, rB, queryL, retrievalL, device, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    topkacc = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        # gnd:一个长度等于gallery数据集的0/1向量，1表示为正样本(即至少包含query的一个标签)，0表示负样本（即，不包含query的标签）。
        gnd = (queryL[iter, :] @ retrievalL.t() > 0).float()
        hamm = CalcHammingDist(qB[iter, :], rB)  # 计算第iter张query图片与gallery数据集的汉明距离
        ind = torch.argsort(hamm)  # 对汉明距离进行排序，返回下标，表示最相似的图片列表
        gnd = gnd[ind]  # 最相似的图片是否为正样本的列表
        tgnd = gnd[0:topk]  # 只看前topk个结果
        tsum = torch.sum(tgnd)  # 前topk个中有多少个预测对了
        if tsum == 0:
            continue
        topkacc += tsum / topk
    topkacc = topkacc / num_query
    torch.cuda.empty_cache()
    return topkacc

def mean_average_precision(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=None,
                           ):
    """
    Calculate mean average precision(map).
    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.
    Returns:
        meanAP (float): Mean Average Precision.
    """
    if topk == None:
        topk = database_labels.shape[0]

    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1, as_tuple=False).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    torch.cuda.empty_cache()
    return mean_AP

#
def mean_average_precision_2(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=None,
                           ):
    """
    Calculate mean average precision(map).
    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.
    Returns:
        meanAP (float): Mean Average Precision.
    """
    if topk == None:
        topk = database_labels.shape[0]

    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1, as_tuple=False).squeeze() + 1.0).float()

        mean_AP += (score / index).sum() / topk

    mean_AP = mean_AP / num_query
    torch.cuda.empty_cache()
    return mean_AP

def precision_k(query_code, database_code, query_labels, database_labels, topk=None):

    num_query = query_labels.shape[0]

    precision_topk = 0

    for iter in range(num_query):
        # gnd:一个长度等于gallery数据集的0/1向量，1表示为正样本(即至少包含query的一个标签)，0表示负样本（即，不包含query的标签）。
        gnd = (query_labels[iter, :] @ database_labels.t() > 0).float()
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[iter, :] @ database_code.t())
        # print(hamming_dist)
        # hamm = CalcHammingDist(qB[iter, :], rB)  # 计算第iter张query图片与gallery数据集的汉明距离
        ind = torch.argsort(hamming_dist)  # 对汉明距离进行排序，返回下标，表示最相似的图片列表
        gnd = gnd[ind]  # 最相似的图片是否为正样本的列表
        tgnd = gnd[0:topk]  # 只看前topk个结果
        tsum = tgnd.sum().int().item()  # 前topk个中有多少个预测对了
        if tsum == 0:
            continue
        precision_topk += tsum / topk
    res = precision_topk / num_query
    torch.cuda.empty_cache()
    return res

def precision_k_other(query_code, database_code, query_labels, database_labels, topk=None):

    num_query = query_labels.shape[0]

    precision_topk = 0

    for iter in range(num_query):
        # gnd:一个长度等于gallery数据集的0/1向量，1表示为正样本(即至少包含query的一个标签)，0表示负样本（即，不包含query的标签）。
        gnd = (query_labels[iter, :] @ database_labels.t() > 0).float()
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[iter, :] @ database_code.t())
        # hamm = CalcHammingDist(qB[iter, :], rB)  # 计算第iter张query图片与gallery数据集的汉明距离
        # ind = torch.argsort(hamming_dist) # 对汉明距离进行排序，返回下标，表示最相似的图片列表
        ind = torch.sort(hamming_dist,stable=True)[1] # 对汉明距离进行排序，返回下标，表示最相似的图片列表
        gnd = gnd[ind]  # 最相似的图片是否为正样本的列表
        tgnd = gnd[0:topk]  # 只看前topk个结果
        tsum = tgnd.sum().int().item()  # 前topk个中有多少个预测对了
        if tsum == 0:
            continue
        precision_topk += tsum / topk
    res = precision_topk / num_query
    torch.cuda.empty_cache()
    return res

def recall_k(query_code, database_code, query_labels, database_labels, topk=None):

    num_query = query_labels.shape[0]

    recall_topk = 0

    for iter in range(num_query):
        # gnd:一个长度等于gallery数据集的0/1向量，1表示为正样本(即至少包含query的一个标签)，0表示负样本（即，不包含query的标签）。
        gnd = (query_labels[iter, :] @ database_labels.t() > 0).float()
        total = gnd.sum().int().item()
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[iter, :] @ database_code.t())
        # hamm = CalcHammingDist(qB[iter, :], rB)  # 计算第iter张query图片与gallery数据集的汉明距离
        ind = torch.argsort(hamming_dist)  # 对汉明距离进行排序，返回下标，表示最相似的图片列表
        gnd = gnd[ind]  # 最相似的图片是否为正样本的列表
        tgnd = gnd[0:topk]  # 只看前topk个结果
        tsum = tgnd.sum().int().item()  # 前topk个中有多少个预测对了
        if tsum == 0:
            continue
        recall_topk += tsum / total
    res = recall_topk / num_query
    torch.cuda.empty_cache()
    return res

# 根据 topk,从1开始，间隔50个点采样
def pr_curve_1(qF, rF, qL, rL, topK=-1):
    n_query = qF.shape[0]
    if topK == -1 or topK > rF.shape[0]:  # top-K 之 K 的上限
        topK = rF.shape[0]

    Gnd = (qL @ rL.t() > 0).float()
    Rank = torch.argsort(CalcHammingDist(qF,rF))

    P, R = [], []
    for k in range(1, topK + 1, 50):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = torch.zeros(n_query)  # 各 query sample 的 Precision@R
        r = torch.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = torch.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = torch.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k  # 求出所有查询样本的Percision@K
            r[it] = gnd_r / gnd_all  # 求出所有查询样本的Recall@K

        P.append(torch.mean(p))
        R.append(torch.mean(r))

    # 画 P-R 曲线
    # fig = plt.figure(figsize=(5, 5))
    # plt.plot(R, P)  # 第一个是 x，第二个是 y
    # plt.grid(True)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # # plt.legend()
    # plt.show()

    return P,R


# 根据 topk,从1开始，取样策略有调整
def pr_curve_3(qF, rF, qL, rL, topK=-1):
    n_query = qF.shape[0]
    if topK == -1 or topK > rF.shape[0]:  # top-K 之 K 的上限
        topK = rF.shape[0]

    Gnd = (qL @ rL.t() > 0).float()
    Rank = torch.argsort(chunked_hamming_distance(qF,rF))

    P, R = [], []
    for k in range(1, 100 + 1, 20):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = torch.zeros(n_query)  # 各 query sample 的 Precision@R
        r = torch.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            # gnd = (qL[it] @ rL.t() > 0).float()
            gnd_all = torch.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = torch.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k  # 求出所有查询样本的Percision@K
            r[it] = gnd_r / gnd_all  # 求出所有查询样本的Recall@K
        P.append(torch.mean(p))
        R.append(torch.mean(r))

    for k in range(101, 200 + 1, 30):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = torch.zeros(n_query)  # 各 query sample 的 Precision@R
        r = torch.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = torch.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = torch.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k  # 求出所有查询样本的Percision@K
            r[it] = gnd_r / gnd_all  # 求出所有查询样本的Recall@K
        P.append(torch.mean(p))
        R.append(torch.mean(r))

    for k in range(201, 400 + 1, 50):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = torch.zeros(n_query)  # 各 query sample 的 Precision@R
        r = torch.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = torch.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = torch.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k  # 求出所有查询样本的Percision@K
            r[it] = gnd_r / gnd_all  # 求出所有查询样本的Recall@K
        P.append(torch.mean(p))
        R.append(torch.mean(r))

    for k in range(401, topK + 1, 200):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = torch.zeros(n_query)  # 各 query sample 的 Precision@R
        r = torch.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = torch.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = torch.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / k  # 求出所有查询样本的Percision@K
            r[it] = gnd_r / gnd_all  # 求出所有查询样本的Recall@K
        P.append(torch.mean(p))
        R.append(torch.mean(r))

    # 画 P-R 曲线
    # fig = plt.figure(figsize=(5, 5))
    # plt.plot(R, P, marker='o', linewidth=1)  # 第一个是 x，第二个是 y
    # plt.grid(True)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # # plt.legend()
    # plt.show()

    return P,R

#  根据汉明距离， 遍历, 慢
def pr_curve_HammingDist(qF, rF, qL, rL, hash_size):
    n_query = qF.shape[0]

    Gnd = (qL @ rL.t() > 0).float()

    hammingdist = CalcHammingDist(qF,rF)
    Rank = torch.argsort(hammingdist)

    P, R = [], []
    for k in range(0, hash_size + 1):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = torch.zeros(n_query)  # 各 query sample 的 Precision@R
        r = torch.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = torch.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue

            for j in range(Rank.shape[1]):
                if hammingdist[it][Rank[it][j]] <= k:
                    continue

            asc_id = Rank[it][:j]
            gnd = gnd[asc_id]
            gnd_r = torch.sum(gnd)  # top-K 中的相关样本数

            p[it] = gnd_r / j  # 求出所有查询样本的Percision@K
            r[it] = gnd_r / gnd_all  # 求出所有查询样本的Recall@K

        P.append(torch.mean(p))
        R.append(torch.mean(r))

    # 画 P-R 曲线
    fig = plt.figure(figsize=(5, 5))
    plt.plot(R, P)  # 第一个是 x，第二个是 y
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('recall')
    plt.ylabel('precision')
    # plt.legend()
    plt.show()

    return P,R

# 汉明距离  where， 分母 1e-8
def pr_curve_HammingDist_1(qF, rF, qL, rL, hash_size):
    n_query = qF.shape[0]

    Gnd = (qL @ rL.t() > 0).float()

    hammingdist = CalcHammingDist(qF,rF)

    P, R = [], []
    for k in range(0, hash_size + 1):  # 枚举 top-K 之 K
        # ground-truth: 1 vs all
        p = torch.zeros(n_query)  # 各 query sample 的 Precision@R
        r = torch.zeros(n_query)  # 各 query sample 的 Recall@R
        for it in range(n_query):  # 枚举 query sample
            gnd = Gnd[it]
            gnd_all = torch.sum(gnd)  # 整个被检索数据库中的相关样本数
            if gnd_all == 0:
                continue

            indices = torch.where(hammingdist[it] <= k)[0]
            # if len(indices) == 0: # 满足距离的个数==0
            #     continue
            # asc_id = Rank[it][:j]
            gnd = gnd[indices]
            gnd_r = torch.sum(gnd)  # top-K 中的相关样本数
            p[it] = gnd_r / (len(indices) +1e-8) # 求出所有查询样本的Percision@K
            r[it] = gnd_r / gnd_all  # 求出所有查询样本的Recall@K

        P.append(torch.mean(p))
        R.append(torch.mean(r))
        print(k)

    # 画 P-R 曲线
    fig = plt.figure(figsize=(5, 5))
    plt.plot(R, P)  # 第一个是 x，第二个是 y
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('recall')
    plt.ylabel('precision')
    # plt.legend()
    plt.show()

    return P,R

# topk numpy
def pr_curve_2(qF, rF, qL, rL, draw_range):
    n_query = qF.shape[0]  # 多少个查询，3
    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    '''
    print(Gnd)  #是一个3行7列的数组。第一行代表gallery的7个元素是否与query[0]同类；第二行代表gallery的7个元素是否与query[1]同类。。。
    [[0. 1. 1. 0. 0. 0. 1.]    #gallery[0]与query[0]不同类；gallery[1]与query[0]同类；gallery[2]与query[0]同类；gallery[3]与query[0]不同类。。。
     [1. 1. 1. 0. 1. 0. 1.]
     [0. 0. 1. 1. 0. 1. 0.]]
    '''

    Rank = np.argsort(CalcHammingDist_numpy(qF, rF))  # 是一个3行7列的数组。
    '''
    print(Rank)             #是一个3行7列的数组。
    [[3 0 2 5 1 4 6]        #gallery的七个元组中，与query[0]最近的元素是gallery[3]，其次是gallery[0]，再次是gallery[2]。。。
     [6 1 4 0 2 5 3]
     [0 4 2 5 6 1 3]]
    '''

    P, R = [], []

    for k in range(1,draw_range+1):  # 比如k=5
        p = np.zeros(n_query)  # [0, 0, 0]  分别是query[0]的acc&k, query[1]的acc&k, query[2]的acc&k
        r = np.zeros(n_query)  # [0, 0, 0]  分别是query[0]的recall&k, query[1]的recall&k, query[2]的recall&k
        for it in range(n_query):  # 比如it=0
            gnd = Gnd[it]  # [0. 1. 1. 0. 0. 0. 1.]
            gnd_all = np.sum(gnd)  # 3，为了求召回率
            if gnd_all == 0:  # 如果没有对的，那准确率和召回率肯定都是0,不用继续求了
                continue
            asc_id = Rank[it][:k]  # [3 0 2 5 1]
            gnd = gnd[asc_id]  # [0 0 1 0 1]
            gnd_r = np.sum(gnd)  # 前k个结果中对了2个。
            p[it] = gnd_r / k  # 准确率：2/5
            r[it] = gnd_r / gnd_all  # 召回率：2/3
        P.append(np.mean(p))
        R.append(np.mean(r))

    # 绘制PR曲线
    plt.plot(R, P, linestyle="-", marker='D', label="DSH")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend()  # 加图例

    plt.show()

    return P, R


if __name__ == '__main__':
    queryBinary = np.array([[1, -1, 1, 1], [-1, 1, -1, -1], [1, -1, -1, -1]])
