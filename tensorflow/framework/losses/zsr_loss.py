import tensorflow as tf

from ..losses import factory as losses
from ..utilities.proxy_utils import ProxyLabelInitializer

from ..configs import default
from ..configs.config import CfgNode as CN

from . import register_loss

# zero_shot_augmented loss configs
# ==================================
zsr = CN()
zsr.function = 'proxy_nca'
zsr.zero_shot_prediction = True
zsr.prediction_loss_weight = 0.1
zsr.regression_regularizer_weight = 0.05

default.cfg.loss.zsr = zsr

@register_loss(name='zsr')
class ZeroShotRegularizedLoss(tf.keras.losses.Loss):

    def __init__(self, model, cfg, **kwargs):
        if 'name' in kwargs.keys():
            name = kwargs['name']
        else:
            name = 'zsr'

        super(ZeroShotRegularizedLoss, self).__init__(
            reduction=tf.keras.losses.Reduction.NONE, name=name)

        class LossWithZSR(getattr(losses, cfg.loss.zsr.function)):
            def __init__(self, model, cfg, **kwargs):

                self.zero_shot_prediction = cfg.loss.zsr.zero_shot_prediction
                self.prediction_loss_weight = cfg.loss.zsr.prediction_loss_weight
                self.regression_regularizer_weight = cfg.loss.zsr.regression_regularizer_weight

                self.softmax_scale = 9.5
                self.softmax_margin = 0.00

                proxy_per_class = 1
                num_proxy = model.num_classes * proxy_per_class

                self._embedding_size = cfg.model.embedding_head.embedding_size

                self.meta_proxy = model.arch.add_weight(name='{}_loss/zsr_proxy'.format(cfg.loss.zsr.function),
                                                        shape=(num_proxy, self._embedding_size),
                                                        dtype=tf.float32,
                                                        initializer=tf.keras.initializers.he_normal,
                                                        trainable=True)



                label_initializer = ProxyLabelInitializer(proxy_per_class)
                self.meta_labels = model.arch.add_weight(name='{}_loss/zsr_label'.format(cfg.loss.zsr.function),
                                                                 shape=(num_proxy, 1),
                                                                 dtype=tf.int32,
                                                                 initializer=label_initializer,
                                                                 trainable=False)

                self._model = model.arch
                self._support_size = cfg.model.embedding_head.GeneralizedSumPooling.support_size

                if not self.zero_shot_prediction:
                    self._regressor = self._model.add_weight(
                        name='attribute_regressor',
                        shape=(self._support_size, self._embedding_size),
                        dtype=tf.float32,
                        initializer=tf.keras.initializers.HeNormal(),  # tf.keras.initializers.zeros
                        trainable=True)

                super(LossWithZSR, self).__init__(model=model, cfg=cfg, **kwargs)

            @tf.function
            def predictior(self, train_samples, train_targets, test_samples):
                x = train_samples
                y = train_targets

                num_samples = tf.shape(train_samples)[1]

                # linear predictor (ie y = xw)
                xxT = tf.matmul(x, x, transpose_b=True)
                reg = self.regression_regularizer_weight * tf.eye(num_samples, batch_shape=[1])
                inv_xxT_plus_reg = tf.linalg.inv(xxT + reg)
                # optimal predictior xT inv(xxT + lambda I) y
                w = tf.matmul(tf.matmul(x, inv_xxT_plus_reg, transpose_a=True), y)

                # now predict embeddings
                x_emb = tf.matmul(test_samples, w)

                # reconstruction error
                '''def debug_print():
                    loss_recon = tf.reduce_mean(tf.reduce_sum(tf.square(tf.matmul(x, w) - y), axis=-1))
                    tf.print('\nreconstruction error: ', loss_recon, '\n')
                    return self.global_step

                def no_print():
                    return self.global_step

                step = tf.cond(tf.equal(tf.math.mod(self.global_step, 10), 0),
                               true_fn=debug_print,
                               false_fn=no_print)'''

                #recon_err = tf.reduce_mean(tf.reduce_sum(tf.square(tf.matmul(x, w) - y), axis=-1))
                #tf.summary.scalar('recon_err', recon_err, step=self.global_step)

                return tf.reshape(x_emb, shape=(-1, self._embedding_size))

            @tf.function
            def train_predictor(self, x_attr, labels):

                y = tf.nn.embedding_lookup(
                    self.meta_proxy,
                    tf.cast(tf.squeeze(labels), dtype=tf.int32)
                )

                x_1 = x_attr[::2]
                y_1 = y[::2]

                x_2 = x_attr[1::2]
                y_2 = y[1::2]

                train_samples = tf.stack([x_2, x_1], axis=0)
                train_targets = tf.stack([y_2, y_1], axis=0)
                test_samples = tf.stack([x_1, x_2], axis=0)

                x_emb = self.predictior(train_samples, train_targets, test_samples)

                lbls = tf.concat([labels[::2], labels[1::2]], axis=0)

                # prediction error
                '''def debug_print():
                    pred_err = tf.reduce_mean(tf.reduce_sum(tf.square(x_emb - tf.concat([y_1, y_2], axis=0)), axis=-1))
                    tf.print('prediction error: ', pred_err, '\n')
                    return self.global_step.assign_add(1)

                def no_print():
                    return self.global_step.assign_add(1)

                step = tf.cond(tf.equal(tf.math.mod(self.global_step, 10), 0),
                               true_fn=debug_print,
                               false_fn=no_print)'''
                # tf.summary.scalar('pred_err', pred_err, step=self.global_step)

                return x_emb, lbls

            @tf.function
            def attribute_loss(self, y_true, y_pred):
                labels, embeddings = y_true, y_pred
                ref_embeddings = self.meta_proxy
                ref_labels = self.meta_labels

                '''if self.normalize_embeddings:
                    embeddings = self.l2normalization(embeddings)
                    ref_embeddings = self.l2normalization(ref_embeddings)'''

                cos_sims = tf.matmul(ref_embeddings, embeddings, transpose_b=True)

                pos_pair_mask = tf.stop_gradient(tf.cast(tf.equal(ref_labels, tf.transpose(labels)), tf.float32))

                logits = self.softmax_scale * (cos_sims - self.softmax_margin * pos_pair_mask)

                prob_of_nhood = tf.nn.softmax(logits, axis=0)

                expected_num_pos = tf.reduce_sum(prob_of_nhood * pos_pair_mask, axis=0) + 1.e-16

                xent = - tf.math.log(expected_num_pos)

                loss = tf.reduce_mean(xent)

                return loss


            def call(self, y_true, y_pred):

                labels = y_true

                x_pooled = y_pred[:, :self._embedding_size]
                x_attribute = y_pred[:, self._embedding_size:]

                if self.zero_shot_prediction:
                    x_pred, lbls = self.train_predictor(x_attribute, labels)
                    l_prediction = self.attribute_loss(lbls, x_pred)
                else:
                    y_pred_emb = x_attribute @ self._regressor
                    l_prediction = self.attribute_loss(y_true, y_pred_emb)

                l_dml = super(LossWithZSR, self).call(y_true, x_pooled)

                total_loss = tf.add_n([
                    (1.0 - self.prediction_loss_weight) * l_dml,
                    self.prediction_loss_weight * l_prediction
                ])

                return total_loss

        self.loss_fn = LossWithZSR(model=model,
                                   cfg=cfg,
                                   **kwargs)

    def call(self, y_true, y_pred):

        return self.loss_fn.call(y_true, y_pred)
