import torch
from torch import nn
import torch.nn.functional as F

from .utils import weights_init_kaiming, weights_init_classifier

class LinearNorm(nn.Module):
    def __init__(self, in_channels, emb_dim, num_classes):
        super(LinearNorm, self).__init__()

        self.embed = nn.Linear(in_channels, emb_dim)
        self.embed.apply(weights_init_kaiming)

        self.classifier = nn.Linear(emb_dim, num_classes)
        self.classifier.apply(weights_init_kaiming)

    def forward(self, features):
        x_gmp = F.adaptive_max_pool2d(features, output_size=1)
        x_gap = F.adaptive_avg_pool2d(features, output_size=1)
        x = 0. * x_gap + x_gmp
        x = x.view(x.size(0), -1)
        x = self.embed(x)
        x_probs = self.classifier(x)
        x_emb = nn.functional.normalize(x, p=2, dim=1)
        x_attr = 0. * x_emb
        return x_probs, x_emb, x_attr
