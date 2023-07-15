import torch
from torch import nn
import torch.nn.functional as F

from ..utils import weights_init_kaiming, weights_init_classifier

from ..gsp.norm_layer import LipschitzL2Normalization
from ..gsp.pdist_layer import PDistL2
from ..gsp.transport_layer import PartialTransportPlan

class GSP(nn.Module):
    def __init__(self, in_channels, emb_dim, num_classes):
        super(GSP, self).__init__()

        self.embedding_size = emb_dim
        self.input_feat_size = in_channels
        self.num_classes = num_classes

        self.num_protos = 64 if num_classes < 1000 else 128
        self.transport_ratio = 0.3
        self.entropy_regularizer_weight = 0.1
        self.optimization_steps = 100

        self.pooling_method = 'gmp'

        # inheriting from typical embedding layer (i.e. embed and classify)
        self.embed = nn.Linear(in_channels, emb_dim)
        self.embed.apply(weights_init_kaiming)

        self.classifier = nn.Linear(emb_dim, num_classes)
        self.classifier.apply(weights_init_kaiming)

        # normalization
        self.normalize = LipschitzL2Normalization.apply

        # distance layer
        self.compute_pdists = PDistL2()

        # transport layer
        self.prototypes = nn.Parameter(
            data=torch.nn.init.kaiming_normal_(
                torch.empty(size=(self.num_protos, self.embedding_size)),
                a=0, mode='fan_out'),
            requires_grad=True)
        self.is_proto_inited = nn.Parameter(data=torch.tensor(False, dtype=torch.bool),
                                            requires_grad=False)  # must be initialized
        self.compute_transport_plan = PartialTransportPlan.apply

        # generalized mean pooling
        if self.pooling_method == 'gmeanp':
            init_power = 3.0
            power_shape = (1, 1, 1, 1)
            learn_power = True
            self.power = nn.Parameter(
                data=torch.full(size=power_shape,
                                fill_value=init_power,
                                dtype=torch.float32),
                requires_grad=learn_power)


    def init_protos(self):
        with torch.no_grad():
            self.is_proto_inited = nn.Parameter(data=torch.tensor(True, dtype=torch.bool),
                                                requires_grad=False)

            global_set = self.classifier.weight

            p = torch.unsqueeze(global_set[0], dim=0)

            point_set = global_set[1:]

            min_dists = torch.square(p - point_set).sum(dim=-1)

            s_cover = [p]

            # greedy k-center
            # 1) get the points of maximum min-distance
            # 2) remove picked points from representative pool (make distance zero)
            # 3) compute distances between picked samples and the point set
            # 4) update min-distances, ie. min(mindist,newdists)
            # 5) return 1 until K many points are picked
            for k in range(self.num_protos - 1):
                # 1)
                argmax = torch.argmax(min_dists)
                p = torch.unsqueeze(point_set[argmax], dim=0)

                # 2)
                # note that 2 is implicitly performed during 3 and 4

                # 3)
                dists = torch.square(p - point_set).sum(dim=-1)

                # 4)
                min_dists = torch.minimum(min_dists, dists)

                # 5)
                s_cover.append(p)

            self.prototypes = nn.Parameter(
                data=torch.cat(s_cover, dim=0),
                requires_grad=True)

    def gap(self, features):
        x_pooled = F.adaptive_avg_pool2d(features,
                                         output_size=1)
        x_pooled = x_pooled.view(x_pooled.size(0), -1)
        x_probs = self.classifier(x_pooled)
        x_emb = nn.functional.normalize(x_pooled, p=2, dim=1)

        return x_probs, x_emb

    def gmp(self, features):

        x_gmp = F.adaptive_max_pool2d(features, output_size=1)
        x_gmp = x_gmp.view(x_gmp.size(0), -1)
        x_emb_un = self.embed(x_gmp)
        x_probs = self.classifier(x_emb_un)
        x_emb = nn.functional.normalize(x_emb_un, p=2, dim=1)

        return x_probs, x_emb

    def gmeanp(self, features):

        x = torch.maximum(features, torch.tensor(1.0e-6))

        x_to_p = torch.pow(x, self.power)

        x_to_p_avg = F.adaptive_avg_pool2d(x_to_p, output_size=1)

        x_pooled = torch.pow(x_to_p_avg, 1.0 / self.power)

        x_pooled = x_pooled.view(x_pooled.size(0), -1)

        x_emb = self.embed(x_pooled)

        x_probs = self.classifier(x_emb)

        x_emb_normalized = nn.functional.normalize(x_emb, p=2, dim=1)

        return x_probs, x_emb_normalized

    def forward(self, features):

        if not self.is_proto_inited:
            self.init_protos()

        # per pixel embedding transform
        bow_feats = torch.nn.functional.conv2d(
            input=features,
            weight=self.embed.weight.unsqueeze(-1).unsqueeze(-1),
            bias=self.embed.bias)

        # pairwise distance (binary costs)
        c_b = self.compute_pdists(
            self.normalize(self.prototypes),
            self.normalize(bow_feats.permute(0, 2, 3, 1))
        )

        P = self.compute_transport_plan(c_b,
                                        self.transport_ratio,
                                        self.entropy_regularizer_weight,
                                        self.optimization_steps)

        mu = self.transport_ratio

        # transport kernel
        ker_tpd = P[:, 1:] / mu


        # simpler computation using residual masses
        num_feats = features.shape[-2] * features.shape[-1]
        rho = P[:, 0].unsqueeze(1)  # residual masses
        mixing_weights = (1. - num_feats * rho) / mu

        if self.pooling_method == 'gmeanp':
            x_probs, x_emb = self.gmeanp(mixing_weights * features)
        elif self.pooling_method == 'gmp':
            x_probs, x_emb = self.gmp(mixing_weights * features)
        elif self.pooling_method == 'gap':
            x_probs, x_emb = self.gap(mixing_weights * bow_feats)
        else:
            x_probs, x_emb = self.gap(mixing_weights * bow_feats)

        # attribute embedding
        x_attr = ker_tpd.sum(dim=(-2, -1))

        return x_probs, x_emb, x_attr

