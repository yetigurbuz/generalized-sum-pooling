import torch
from torch import nn

class PDistL2(nn.Module):
    def __init__(self, squared: bool = False):
        super(PDistL2, self).__init__()

        self.squared = squared


    def forward(self, x, y):
        """Computes the pairwise distance matrix with numerical stability.
            assuming x has batch of M vectors of D-dim and y has N vectors of the same dimension
             returns pdist_kij = ||x_ki - y_j||_2
             or similarly if y has batch of N vectors
             returns  pdist_kij = ||x_i - y_kj||_2
                    Args:
                      x: 2(3)-D Tensor of size [(batch size), number of data, feature dimension].
                      y: 2(3)-D Tensor of size [(batch size), number of data, feature dimension].

                      squared: Boolean, whether or not to square the pairwise distances.
                    Returns:
                      pairwise_distances: 3-D Tensor of size [batch size, number of data, number of data].

            This function also supports when x is of shape=(batch, height, width, feature_size)
             and y is of shape=(num_features, feature_size) or vice-versa
             NOTE: we always assume both x and y has the last axis as feature axis i.e., -1
                    """

        #x = features[0]
        #y = features[1]

        shape_x = x.shape
        shape_y = y.shape

        rank_x = len(shape_x)
        rank_y = len(shape_y)


        out_shape = None

        if rank_x > rank_y:
            if rank_x > 3:
                x = torch.reshape(x, shape=(-1, shape_x[-1]))
                if rank_y > 2:
                    y = torch.reshape(y, shape=(-1, shape_y[-1]))
                    out_shape = torch.Size(shape_x[:-1] + shape_y[:-1])
                else:
                    out_shape = torch.Size(shape_x[:-1] + (shape_y[0], ))
        elif rank_y > rank_x:
            # we assume first dimension of the greater rank tensor is the batch dimension
            if rank_y > 3:
                y = torch.reshape(y, shape=(shape_y[0], -1, shape_y[-1]))
                if rank_x > 2:
                    x = torch.reshape(x, shape=(-1, shape_x[-1]))
                    out_shape = torch.Size((shape_y[0], ) + shape_x[:-1] + shape_y[1:-1])
                else:
                    out_shape = torch.Size((shape_y[0], shape_x[0]) + shape_y[1:-1])
        else:
            if rank_x > 2:
                x = torch.reshape(x, shape=(-1, shape_x[-1]))
                y = torch.reshape(y, shape=(-1, shape_y[-1]))
                out_shape = torch.Size(shape_x[:-1] + shape_y[:-1])

        pairwise_distance_matrix = torch.cdist(x, y)

        if self.squared:
            pairwise_distance_matrix = torch.square(pairwise_distance_matrix)

        if out_shape is not None:
            pairwise_distance_matrix = torch.reshape(pairwise_distance_matrix, shape=out_shape)

        return pairwise_distance_matrix
