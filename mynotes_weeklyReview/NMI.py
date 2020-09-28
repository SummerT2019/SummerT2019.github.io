import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np

# mutual information base neural estimation
def mi_theta(net, x, y):
    return net(x).mean() - net(y).exp().mean().log()

def net_mi(x, y):
    iter_num = 150
    net = nn.Sequential(
        nn.Linear(16, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    trainer = optim.Adam(net.parameters(), lr=0.01)


    for j in range(iter_num):
        if j != iter_num-1:
            #print(x)
            #print(y)
            sample_mi = -mi_theta(net, x, y)
            trainer.zero_grad()
            sample_mi.backward(retain_graph=True)
            trainer.step()
        else:
            sample_mi = -mi_theta(net, x, y)
            trainer.zero_grad()
            return sample_mi



'''
# mutual information base numpy
def NMI(A, B):
    # 样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # 互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:  # 遍历所有的元素对　
        for idB in B_ids:
            # idA index
            idAOccur = np.where(A == idA)
            # idB index
            idBOccur = np.where(B == idB)
            # idA-idB index
            idABOccur = np.intersect1d(idAOccur, idBOccur)
            # count/total = p 计算边缘概率分布和联合概率分布
            px = 1.0 * len(idAOccur[0]) / total
            py = 1.0 * len(idBOccur[0]) / total
            pxy = 1.0 * len(idABOccur) / total
            # 计算互信息
            MI = MI + pxy * math.log(pxy / (px * py) + eps, 2)
    # 计算ｘ和ｙ熵
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0 * len(np.where(A == idA)[0])
        Hx = Hx - (idAOccurCount / total) * math.log(idAOccurCount / total + eps, 2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0 * len(np.where(B == idB)[0])
        Hy = Hy - (idBOccurCount / total) * math.log(idBOccurCount / total + eps, 2)
    # 标准化互信息
    MI_std = 2.0 * MI / (Hx + Hy)
    return MI


def NMI_2D(array1, array2):
    NMI_SUM = np.empty(shape=(array1.shape[0], 1))
    for i in range(array1.shape[0]):
        NMI_SUM[i][0] = NMI(array1[i], array2[i])
    return NMI_SUM.mean()


if __name__ == '__main__':
    a = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    b = np.array([1, 2, 1, 1, 4, 1, 2, 2, 6, 2, 2, 2, 3, 7, 3, 3, 3])
    A = np.array([a, a, a])  # 创建6行3列的二维数组
    B = np.array([b, b, b])  # 创建6行3列的二维数组
    print(NMI(a, b))
    print(NMI_2D(A, B))
'''
