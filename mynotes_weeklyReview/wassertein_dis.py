from __future__ import print_function
from scipy.stats import wasserstein_distance
import numpy as np
from tools import emd_layers
import torch

x_test = np.array([[-0.1606, 0.7606, -0.1606, 0.1730, 0.1493, 0.2491, -0.1606, -0.1606, -0.1606, -0.1606],
                  [-0.1049, 0.8766, -0.1049, 0.2014, 0.1547, 0.1884, -0.1049, -0.1049, -0.1049, -0.1049],
                  [-0.1219, 0.8047, -0.1088, 0.2379, 0.2071, 0.2929, -0.1219, -0.1219, -0.1219, -0.0997],
                  [-0.1026, 0.8289, -0.0824, 0.1022, 0.3252, 0.2723, -0.1026, -0.1026, -0.1026, -0.1026],
                  [-0.1446, 0.8303, -0.1446, 0.1689, 0.0769, 0.1754, -0.1446, -0.1244, -0.1446, -0.1446],
                  [-0.1522, 0.7899, -0.1522, 0.1758, 0.2221, 0.1336, -0.1522, -0.1522, -0.1522, -0.1522],
                  [-0.1296, 0.6024, -0.1296, 0.3211, 0.4410, 0.3800, -0.1296, -0.1296, -0.1296, -0.1233],
                  [-0.1182, 0.8582, -0.1182, 0.2294, 0.1185, 0.1705, -0.1182, -0.1182, -0.1182, -0.1182],
                  [-0.1257, 0.7778, -0.1257, 0.3564, 0.2777, 0.1502, -0.1257, -0.1257, -0.1257, -0.0814]])
y_test = np.array([[-0.0916, -0.0916, -0.0916, 0.6997, -0.0916, 0.2406, 0.4214, 0.0274, 0.0499, -0.0916],
                   [-0.2627, -0.2627, -0.2627, 0.2805, 0.0675, 0.2603, -0.0357, -0.2056,  0.3779, -0.2627],
                   [-0.0443, -0.0443, 0.0130, 0.4030, -0.0443, 0.1078, 0.7448, 0.1099, -0.0069, 0.0234],
                   [-0.2950, -0.2950, -0.2166, 0.3417, 0.1545, 0.2188, 0.1492, -0.1070, -0.1006, -0.2950],
                   [-0.2113, -0.2113, -0.2113, 0.6830, -0.2113, -0.1363, -0.2113, -0.2113, 0.1341, -0.0031],
                   [-0.1988, -0.1988, -0.1968, 0.5014, -0.1959, 0.1446, -0.0003, 0.0586, -0.0136, -0.1988],
                   [-0.2489, -0.2489, -0.0215, 0.5107, -0.1582, 0.0281, -0.1240, 0.2930, -0.2489, -0.2348],
                   [-0.1883, -0.0304, -0.1418, -0.1883, 0.7764, -0.1883, 0.0867, -0.1883, -0.1883, -0.0098],
                   [-0.1666, -0.1666, -0.1666, 0.5500, -0.1666, -0.0068, 0.6710, 0.1456, -0.0438, 0.0423]])
np.random.seed(42)

n_points = 5
a = np.array([[i, 0] for i in range(n_points)])
b = np.array([[i, 1] for i in range(n_points)])
x = torch.tensor(a, dtype=torch.float)
y = torch.tensor(b, dtype=torch.float)

# em　based numpy
def EM_dis(array1, array2):
    em_dis = np.empty(shape=(array1.shape[0], 1))
    for i in range(array1.shape[0]):
        em_dis[i][0] = wasserstein_distance(array1[i], array2[i])
    return em_dis.mean()

# em based pytorch-Sinkorn
def was(x, y):
    was_dis = emd_layers.SinkhornDistance(eps=0.1, max_iter=200, reduction='mean')
    dist, p, c = was_dis(x, y)
    return dist

def main():
    '''
    x = torch.autograd.Variable(torch.from_numpy(x_test).type(torch.FloatTensor), requires_grad=True)
    y = torch.autograd.Variable(torch.from_numpy(y_test).type(torch.FloatTensor), requires_grad=True)
    print(x.dim())
    print(y.dim())
    '''
    cost, p, c = was(x, y)
    print(cost)

if __name__ == "__main__":
    main()