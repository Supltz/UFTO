import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist, squareform
import torch.nn.functional as F

# HSIC function
def hsic(X, Y, sigma=1.0):
    n = X.shape[0]
    K = rbf_kernel(X, X, gamma=1.0 / (2 * sigma ** 2))
    L = rbf_kernel(Y, Y, gamma=1.0 / (2 * sigma ** 2))
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    hsic_value = (1 / (n - 1) ** 2) * np.trace(Kc @ Lc)
    return hsic_value

class MINE(nn.Module):
    """
    Network for estimating mutual information between two high-dimensional embeddings (e.g., x and z).
    """

    def __init__(self, x_dim, z_dim):
        super().__init__()
        self._layers = nn.ModuleList()
        self._layers.append(nn.Linear(x_dim + z_dim, 512))  # Input is concatenation of x and z
        self._layers.append(nn.Linear(512, 512))
        self._out_layer = nn.Linear(512, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for (name, param) in self._layers.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.zeros_(param)

        for (name, param) in self._out_layer.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, nonlinearity='linear')
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, z):
        """
        Forward pass through the MINE network with two embeddings, x and z.
        """
        combined_input = torch.cat([x, z], dim=1)  # Concatenate x and z
        for hid_layer in self._layers:
            combined_input = F.relu(hid_layer(combined_input))
        return self._out_layer(combined_input)

def mine_loss(mine, x, z):
    """
    Mutual Information loss function using MINE.
    x: embedding 1 (batch_size, 512)
    z: embedding 2 (batch_size, 512)
    """
    # Joint distribution (x, z)
    t_joint = mine(x, z)
    
    # Marginal distribution (x, permuted z)
    z_perm = z[torch.randperm(z.size(0))]
    t_marginal = mine(x, z_perm)
    
    # Use numerically stable log-sum-exp for marginal term
    mi_estimation = torch.mean(t_joint) - torch.log(torch.mean(torch.exp(t_marginal)))
    
    return abs(mi_estimation) * 5  # Return MI estimation

# Distance Correlation function
def distance_correlation(X, Y):
    n = X.shape[0]
    A = squareform(pdist(X, 'euclidean'))
    B = squareform(pdist(Y, 'euclidean'))
    A_mean_row = A.mean(axis=1, keepdims=True)
    A_mean_col = A.mean(axis=0, keepdims=True)
    A_mean = A.mean()
    A_centered = A - A_mean_row - A_mean_col + A_mean
    B_mean_row = B.mean(axis=1, keepdims=True)
    B_mean_col = B.mean(axis=0, keepdims=True)
    B_mean = B.mean()
    B_centered = B - B_mean_row - B_mean_col + B_mean
    dcov = np.sum(A_centered * B_centered) / (n * n)
    dvar_x = np.sum(A_centered * A_centered) / (n * n)
    dvar_y = np.sum(B_centered * B_centered) / (n * n)
    dcor = 0
    if dvar_x > 0 and dvar_y > 0:
        dcor = np.sqrt(dcov) / np.sqrt(np.sqrt(dvar_x * dvar_y))
    return dcor