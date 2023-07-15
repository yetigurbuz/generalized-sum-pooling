import tensorflow as tf

from ..utilities.optimal_transport_utils import partialOptimalTransportPlan
from ..utilities.math_utils import pairwiseL2Distance

from ..utilities.proxy_utils import ProxyLabelInitializer

from .l2_normalization import L2Normalization

@tf.keras.utils.register_keras_serializable()
class PartialTransport(tf.keras.layers.Layer):

    def __init__(self,
                 support_size,
                 cost_fn='l2', # l2 or conv
                 normalization='l2',    # or ema or None
                 entropy_regularizer_weight=10,
                 optimization_steps=10,
                 grad_method='inv',
                 name=None,
                 **kwargs):
        super(PartialTransport, self).__init__(name=name, **kwargs)

        self._support_size = support_size

        self._gamma = entropy_regularizer_weight
        self._optimal_transport_unfold = optimization_steps

        self._cost_fn = cost_fn.lower()

        self._normalization = normalization

        self.maybe_normalize = None

        if self._normalization is not None:
            if self._normalization.lower() == 'l2':
                self.maybe_normalize = L2Normalization()
            elif self._normalization.lower() == 'ema':
                self.maybe_normalize = self.buildEMANormalizer()
            else:
                raise ValueError('for normalization l2 and ema are supported but got {}'.format(self._normalization))
        else:
            self.maybe_normalize = tf.keras.layers.Lambda(function=lambda x: x)

        if not grad_method in ['auto', 'inv']:
            raise ValueError('grad_method must be {} but got {}'.format(['auto', 'inv'], grad_method))
        self._grad_method = grad_method

        self._kernel_constraint = None

        # self._loss_weight = loss_weight

        # self._weight_clip = weight_clip
        # self._bigM = bigM

    def build(self, input_shape):

        feat_shape, ratio_shape, unary_costs_shape = input_shape

        self._feat_dim = feat_shape[-1]


        if self._cost_fn == 'l2':
            # target support
            self._target_support = self.add_weight(shape=(self._support_size, self._feat_dim),
                                                   initializer=tf.keras.initializers.GlorotUniform(),
                                                   constraint=self._kernel_constraint,
                                                   name='{}/target_support'.format(self.name))

        elif 'conv' in self._cost_fn:
            kernel_size = int(self._cost_fn.split('conv')[-1])
            shape = (kernel_size, kernel_size, self._support_size, self._feat_dim)
            self._target_support = self.add_weight(shape=shape,
                                                   initializer=tf.keras.initializers.GlorotUniform(),
                                                   constraint=self._kernel_constraint,
                                                   name='{}/target_support'.format(self.name))

            self._distance_whitening = MovingAverageStandardization(axis=1,
                                                                    name='pdists_normalization')

        else:
            raise ValueError('for binary costs l2 and conv are supported but got {}'.format(self._cost_fn))

        self.kernel = self._target_support


    @tf.function
    def binary_costs(self, features):

        if self._cost_fn == 'l2':

            return pairwiseL2Distance(
                self.maybe_normalize(self._target_support),
                self.maybe_normalize(features)
            )

        elif 'conv' in self._cost_fn:

            conv_out = tf.nn.conv2d(
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

            whitened = conv_out #self._distance_whitening(conv_out)
            return -1.0 * whitened # inverse similarity is distance


        else:
            raise ValueError('for binary costs l2 and conv are supported but got {}'.format(self._cost_fn))

    def call(self, inputs, **kwargs):

        # computes residual masses after transport to the anchors


        features = inputs[0]    # shape [batch_size, width, height, feat_dim]
        transport_ratio = inputs[1]     # shape [batch_size, ]
        c_u = inputs[2]     # shape [batch_size, width, height]

        # broadcastable shapes
        num_expands = len(features.get_shape().as_list()) - 1
        dummy_ones = tf.ones(shape=(tf.shape(features)[0],))
        for _ in range(num_expands):
            transport_ratio = tf.expand_dims(transport_ratio, axis=-1)
            dummy_ones = tf.expand_dims(dummy_ones, axis=-1)
        c_u = tf.expand_dims(c_u, axis=1)


        # pairwise l2 distance: rows => anchors, columns => feature set
        c_b = self.binary_costs(features)

        # compute transport plan
        mu = transport_ratio * dummy_ones
        P = partialOptimalTransportPlan(binary_costs=c_b,
                                        unary_costs=c_u,
                                        mu=mu,
                                        gamma=self._gamma,
                                        max_it=self._optimal_transport_unfold,
                                        grad_method=self._grad_method)

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
        config = super(PartialTransport, self).get_config()
        config.update({'support_size': self._support_size,
                       'cost_fn': self._cost_fn,
                       'normalization': self._normalization,
                       'entropy_regularizer_weight': self._gamma,
                       'optimization_steps': self._optimal_transport_unfold,
                       'grad_method': self._grad_method
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

@tf.keras.utils.register_keras_serializable()
class MovingAverageStandardization(tf.keras.layers.BatchNormalization):

    def __init__(self, name=None, **kwargs):
        super(MovingAverageStandardization, self).__init__(name=name, **kwargs)

    #def call(self, inputs, training=True, **kwargs):
        #return super(MovingAverageStandardization, self).call(inputs, training, **kwargs)
