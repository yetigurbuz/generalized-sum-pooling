import torch
from torch import nn
import torch.nn.functional as F


class ZSRLoss(torch.nn.Module):

    def __init__(self, emb_dim, num_classes):
        super(ZSRLoss, self).__init__()

        self.regression_regularizer_weight = 0.05
        self.softmax_scale = 9.5
        self.softmax_margin = 0.00
        self.num_classes  = num_classes


        self.embedding_size = emb_dim

        self.class_embeddings = nn.Parameter(
            data=torch.nn.init.kaiming_normal_(
                torch.empty(size=(self.num_classes, self.embedding_size)),
                a=0, mode='fan_out'),
            requires_grad=True)

        self.class_labels = nn.Parameter(
            data=torch.arange(self.num_classes,
                              dtype=torch.int32).unsqueeze(-1),
            requires_grad=False)

        self.debug_step = 0
        self.debug_every = 50

    def predictior(self, train_samples, train_targets, test_samples):

        x = train_samples
        y = train_targets

        num_samples = train_samples.shape[1]

        # linear predictor (ie y = xw)
        xxT = torch.matmul(x, x.permute(0, 2, 1))
        reg = self.regression_regularizer_weight * torch.eye(num_samples,
                                                             dtype=torch.float32,
                                                             device=x.device).unsqueeze(0)
        inv_xxT_plus_reg = torch.linalg.inv(xxT + reg)
        # optimal predictior xT inv(xxT + lambda I) y
        w = torch.matmul(torch.matmul(x.permute(0, 2, 1), inv_xxT_plus_reg), y)

        # now predict embeddings
        x_emb = torch.matmul(test_samples, w)

        # reconstruction error
        '''if (self.debug_step % self.debug_every) == 0:
            loss_recon = torch.mean(
                torch.sum(
                    torch.square(
                        torch.matmul(x, w) - y),
                    dim=-1))
            print('\nreconstruction error: {}\n'.format(loss_recon))'''

        return x_emb.reshape(-1, self.embedding_size)

    def train_predictor(self, x_attr, labels):

        y = torch.index_select(self.class_embeddings,
                               dim=0,
                               index=labels.squeeze())

        nb_samples = y.shape[0] // 2

        x_1 = x_attr[:nb_samples]
        y_1 = y[:nb_samples]

        x_2 = x_attr[nb_samples:]
        y_2 = y[nb_samples:]

        train_samples = torch.stack([x_2, x_1], dim=0)
        train_targets = torch.stack([y_2, y_1], dim=0)
        test_samples = torch.stack([x_1, x_2], dim=0)

        x_emb = self.predictior(train_samples, train_targets, test_samples)

        # prediction error
        '''if (self.debug_step % self.debug_every) == 0:
            pred_err = torch.mean(
                torch.sum(
                    torch.square(x_emb - torch.cat([y_1, y_2], dim=0)),
                    dim=-1))
            print('prediction error: {}\n'.format(pred_err))'''

        return x_emb, labels

    def attribute_loss(self, y_true, y_pred):
        labels, embeddings = y_true, y_pred
        ref_embeddings = self.class_embeddings
        ref_labels = self.class_labels

        cos_sims = torch.matmul(ref_embeddings, embeddings.T)

        pos_pair_mask = torch.eq(ref_labels, labels.T).float().detach()

        logits = self.softmax_scale * (cos_sims - self.softmax_margin * pos_pair_mask)

        prob_of_nhood = F.softmax(logits, dim=0)

        expected_num_pos = torch.sum(prob_of_nhood * pos_pair_mask, dim=0) + 1.e-16

        xent = - torch.log(expected_num_pos)

        loss = torch.mean(xent)

        return loss

    def forward(self, x_attribute, labels):

        x_pred, lbls = self.train_predictor(x_attribute, labels)
        l_prediction = self.attribute_loss(lbls, x_pred)

        self.debug_step += 1

        return l_prediction
