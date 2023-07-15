import tensorflow as tf


from ..utilities.math_utils import pairwiseL2Distance


from .l2_normalization import L2Normalization


@tf.keras.utils.register_keras_serializable()
class OptimalTransport(tf.keras.layers.Layer):

    def __init__(self,
                 support_size,
                 cost_fn='l2', # l2 or conv
                 normalization='l2',    # or ema or None
                 entropy_regularizer_weight=10,
                 optimization_steps=10,
                 name=None,
                 **kwargs):
        super(OptimalTransport, self).__init__(name=name, **kwargs)

        self._support_size = support_size

        self._gamma = entropy_regularizer_weight
        self._optimal_transport_unfold = optimization_steps

        self._cost_fn = cost_fn.lower()

        self._normalization = normalization

        self.maybe_normalize = None

        if self._normalization.lower() == 'l2':
            self.maybe_normalize = L2Normalization()
        elif self._normalization.lower() == 'ema':
            self.maybe_normalize = self.buildEMANormalizer()
        elif self._normalization is not None:
            raise ValueError('for normalization l2 and ema are supported but got {}'.format(self._normalization))
        else:
            self.maybe_normalize = tf.keras.layers.Lambda(function=lambda x: x)



        self._kernel_constraint = None

        # self._loss_weight = loss_weight

        # self._weight_clip = weight_clip
        # self._bigM = bigM

    def build(self, input_shape):

        b, w, h, d = input_shape

        self._feat_dim = d

        num_feats = w * h
        self._num_feats = num_feats

        self._flatten = tf.keras.layers.Reshape(target_shape=(self._support_size, self._num_feats))

        self._q = (1. / num_feats) * tf.ones(shape=(1, 1, num_feats))
        self._p = (1. / self._support_size) * tf.ones(shape=(1, 1, self._support_size))


        if self._cost_fn == 'l2':
            # target support
            self._target_support = self.add_weight(shape=(self._support_size, self._feat_dim),
                                                   initializer=tf.keras.initializers.GlorotUniform(),
                                                   constraint=self._kernel_constraint,
                                                   name='{}/target_support'.format(self.name))

        elif self._cost_fn == 'conv':
            self._target_support = self.add_weight(shape=(3, 3, self._support_size, self._feat_dim),
                                                   initializer=tf.keras.initializers.GlorotUniform(),
                                                   constraint=self._kernel_constraint,
                                                   name='{}/target_support'.format(self.name))
        else:
            raise ValueError('for binary costs l2 and conv are supported but got {}'.format(self._cost_fn))

        self.kernel = self._target_support


    def binary_costs(self, features):

        if self._cost_fn == 'l2':

            return pairwiseL2Distance(
                self.maybe_normalize(self._target_support),
                self.maybe_normalize(features)
            )

        elif self._cost_fn == 'conv':

            return - 1.0 * tf.nn.conv2d(
                    input=tf.transpose(
                        self.maybe_normalize(features),
                        perm=[0, 3, 1, 2]),
                    filters=tf.transpose(
                        self.maybe_normalize(self._target_support),
                        perm=[0, 1, 3, 2]),
                    strides=1,
                    padding='SAME',
                    data_format='NCHW'
                )
        else:
            raise ValueError('for binary costs l2 and conv are supported but got {}'.format(self._cost_fn))

    def optimal_transport(self, c):
        # c: cost matrix tensor of shape = (batch_size, num_bins_in_p, num_bins_in_q)

        eps = tf.constant(7. / 3 - 4. / 3 - 1., dtype=tf.float32)
        gamma = tf.constant(self._gamma, dtype=tf.float32)

        K = tf.exp(- gamma * c)

        q = self._q
        p = self._p

        def fixedPointIteration(u):
            # takes u_(n), returns u_(n+1)
            v = q / (tf.matmul(u, K) + eps)

            return p / (tf.matmul(v, K, transpose_b=True) + eps)

        u = tf.ones_like(p)
        for it in range(self._optimal_transport_unfold):
            u = fixedPointIteration(u)

        v = q / (tf.matmul(u, K) + eps)

        Pi = tf.transpose(u, perm=[0, 2, 1]) * K * v

        return Pi


    def call(self, inputs, **kwargs):

        # computes residual masses after transport to the anchors


        features = inputs   # shape [batch_size, width, height, feat_dim]

        # pairwise l2 distance: rows => anchors, columns => feature set
        c_b = self.binary_costs(features)

        c = self._flatten(c_b)

        # compute transport plan
        P = self.optimal_transport(c)

        return P

    def compute_output_shape(self, input_shape):

        input_rank = len(input_shape[0])
        if input_rank > 3:
            out_shape = (input_shape[0][0], self._support_size + 1, input_shape[0][1], input_shape[0][2])
        else:
            out_shape = (input_shape[0][0], self._support_size + 1, input_shape[0][1])

        output_shape = [
            input_shape[0],
            (self._support_size, input_shape[0][-1]),
            out_shape
        ]
        return output_shape

    def get_config(self):
        config = super(OptimalTransport, self).get_config()
        config.update({'support_size': self._support_size,
                       'cost_fn': self._cost_fn,
                       'normalization': self._normalization,
                       'entropy_regularizer_weight': self._gamma,
                       'optimization_steps': self._optimal_transport_unfold
                       }
                      )
        return config

    def buildEMANormalizer(self):
        momentum = 0.9
        self.ema_momented_mean = self.add_weight(shape=(),
                                                 dtype=tf.float32,
                                                 trainable=False,
                                                 initializer=tf.keras.initializers.zeros(),
                                                 constraint=None,
                                                 name='{}/ema_mean'.format(self.name))

        # The variable is used to check whether momented_mean and momented_std are initialized
        self.ema_init = self.add_weight(shape=(),
                                        dtype=tf.int32,
                                        trainable=False,
                                        initializer=tf.keras.initializers.zeros(),
                                        constraint=None,
                                        name='{}/ema_init'.format(self.name))

        def initStatistics(sample_mean):
            momented_mean = self.ema_momented_mean.assign(sample_mean)
            self.ema_init.assign_add(1)
            return momented_mean

        def updateStatistics(sample_mean):
            momented_mean = (1. - momentum) * sample_mean + momentum * self.ema_momented_mean
            momented_mean = self.ema_momented_mean.assign(momented_mean)
            return momented_mean

        def normalization_fn(embeddings):
            norms = tf.stop_gradient(
                tf.sqrt(
                    tf.reduce_sum(
                        tf.square(embeddings),
                        axis=-1, keepdims=True))
            )
            sample_mean = tf.reduce_mean(norms)
            momented_mean = tf.cond(
                pred=tf.equal(self.ema_init, 0),
                true_fn=lambda: initStatistics(sample_mean),
                false_fn=lambda: updateStatistics(sample_mean))

            return embeddings / momented_mean

        return normalization_fn