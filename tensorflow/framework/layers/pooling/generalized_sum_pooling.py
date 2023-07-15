import tensorflow as tf

from ..partial_transport import PartialTransport
from ..l2_normalization import L2Normalization

from ...utilities.proxy_utils import greedyKCenter

from ...configs import default
from ...configs.config import CfgNode as CN

from . import register_pooling

# generalized sum pooling
GeneralizedSumPooling_cfg = CN()
GeneralizedSumPooling_cfg.binary_costs = 'l2'
GeneralizedSumPooling_cfg.normalization = 'l2'
GeneralizedSumPooling_cfg.support_size = 32
GeneralizedSumPooling_cfg.transport_ratio = 0.5
GeneralizedSumPooling_cfg.warm_up_steps = 0
GeneralizedSumPooling_cfg.memory_size = 1024
GeneralizedSumPooling_cfg.entropy_regularizer_weight = 20.0
GeneralizedSumPooling_cfg.optimization_steps = 100
GeneralizedSumPooling_cfg.grad_method = 'inv'   # or auto for autograd
GeneralizedSumPooling_cfg.augment_max_pool = False
GeneralizedSumPooling_cfg.use_gemp = False
GeneralizedSumPooling_cfg.return_attention_maps = False
GeneralizedSumPooling_cfg.image_size = (None, None)

default.cfg.model.embedding_head.GeneralizedSumPooling = GeneralizedSumPooling_cfg

@register_pooling(name='GeneralizedSumPooling')
@tf.keras.utils.register_keras_serializable()
class GeneralizedSumPooling(tf.keras.layers.Layer):

    def __init__(self,
                 embedding_size,
                 support_size,
                 transport_ratio,
                 binary_costs='l2',
                 normalization='l2',  # or ema std or None
                 warm_up_steps=0,  # steps before start prediction of constraints
                 memory_size=1024,
                 entropy_regularizer_weight=20.,
                 optimization_steps=100,
                 grad_method='inv',
                 augment_max_pool=False,
                 use_gemp=False,
                 return_attention_maps=False,
                 image_size=None,
                 name=None,
                 **kwargs):

        super(GeneralizedSumPooling, self).__init__(name=name, **kwargs)

        self._embedding_size = embedding_size
        self._normalization = normalization
        self._binary_costs = binary_costs.lower()
        self._support_size = support_size
        self._transport_ratio = transport_ratio
        self._entropy_regularizer_weight = entropy_regularizer_weight
        self._optimization_steps = optimization_steps
        self._grad_method = grad_method

        self._return_attention_maps = return_attention_maps
        self._img_size = image_size

        self._steps = self.add_weight(shape=(),
                                      dtype=tf.int32,
                                      trainable=False,
                                      initializer=tf.keras.initializers.zeros(),
                                      constraint=None,
                                      name='{}/step_counter'.format(self.name))
        self._warm_up_steps = warm_up_steps

        self._memory_size = memory_size

        self._feat_emb_size = self._embedding_size

        # memory variables
        self._memory = self.add_weight(
            name='{}/emb_memory'.format(self.name),
            shape=(self._memory_size, self._feat_emb_size),
            dtype=tf.float32,
            initializer=tf.keras.initializers.zeros,
            trainable=False)

        self._mem_ptr = self.add_weight(
            name='{}/pointer'.format(self.name),
            shape=(),
            dtype=tf.int32,
            initializer=tf.keras.initializers.zeros,
            trainable=False)

        # The variable is used to check whether memory is initialized
        self._ready = tf.Variable(
            initial_value=tf.constant(False, tf.bool),
            trainable=False, shape=(), dtype=tf.bool)

        self._augment_max_pool = augment_max_pool
        self._use_gemp = use_gemp

    def build(self, input_shape):

        w, h = input_shape[1:3]  # 2D to 1D bag of features representation
        self._num_features = w * h

        self._zeros_cost = tf.keras.models.Sequential(
            [
                tf.keras.layers.Lambda(
                    lambda x: tf.stop_gradient(x)),
                tf.keras.layers.Lambda(
                    lambda x: tf.reduce_sum(0.0 * x, axis=-1))
            ],
            name='zeros_cost'
        )

        self._constant_ratio = tf.keras.models.Sequential(
            [
                tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Lambda(lambda x:
                                       tf.reduce_sum(0.0 * x, axis=-1) + self._transport_ratio),
            ],
            name='constant_ratio'
        )

        # feature transform heads
        self._pre_feature_head = tf.keras.layers.Conv2D(filters=self._feat_emb_size,
                                                        kernel_size=1,
                                                        padding='SAME',
                                                        use_bias=False,
                                                        kernel_initializer='glorot_uniform',
                                                        name='feature_transform')


        # pooling heads
        self._partial_transport = PartialTransport(
            support_size=self._support_size,
            cost_fn=self._binary_costs,
            normalization=self._normalization,
            entropy_regularizer_weight=self._entropy_regularizer_weight,
            optimization_steps=self._optimization_steps,
            grad_method=self._grad_method)

        self._global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self._global_pool_attr = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')

        if self._use_gemp:
            self._power = 3.0
            power_shape = (1, 1, 1, 1)
            self._learn_power = True
            self._p = self.add_weight(
                name="power", shape=power_shape,
                initializer=tf.keras.initializers.constant(
                    tf.ones(shape=power_shape) * self._power),
                trainable=self._learn_power)
            self._gap_keep = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
            self._reshape_to_emb = tf.keras.layers.Reshape(target_shape=[self._embedding_size])


    @tf.function
    def global_pool(self, inputs, training):

        bow_feats = self._pre_feature_head(inputs)

        x_emb = self._global_pool(bow_feats)

        if training:
            self._steps.assign_add(1)

        is_memory_ready = tf.cond(
            pred=tf.greater(self._steps, tf.constant(self._warm_up_steps)),
            true_fn=lambda: self.fill_memory(x_emb),
            false_fn=lambda: self._ready)

        tf.cond(
            pred=tf.equal(is_memory_ready, tf.constant(True)),
            true_fn=self.initialize_support_kernel,
            false_fn=lambda: self._partial_transport.kernel
        )

        zeros_attr = tf.zeros(shape=(tf.shape(x_emb)[0], self._support_size))

        emb = tf.concat([x_emb, zeros_attr], axis=-1)

        return emb

    @tf.function
    def gmeanp(self, inputs, training):

        x = tf.math.maximum(inputs, 1e-6)

        x_to_p = tf.pow(x, self._p)

        x_to_p_avg = self._gap_keep(x_to_p)

        x_pooled = tf.pow(x_to_p_avg, 1.0 / self._p)

        x_emb = self._reshape_to_emb(self._pre_feature_head(x_pooled))

        return x_emb

    @tf.function
    def transport_pool(self, inputs, training):

        bow_feats = self._pre_feature_head(inputs)

        unary_costs = self._zeros_cost(bow_feats)
        transport_ratio = self._constant_ratio(bow_feats)

        P = self._partial_transport([bow_feats, transport_ratio, unary_costs])


        mu = tf.stop_gradient(transport_ratio)
        num_expands = len(P.get_shape().as_list()) - 1
        for _ in range(num_expands):
            mu = tf.expand_dims(mu, axis=-1)

        # transport kernel
        ker_tpd = P[:, 1:] / mu

        # ensemble way of computation of pooled features
        '''set_of_pooled = (tf.reshape(ker_tpd, shape=(-1, self._support_size, self._num_features)) @
                         tf.reshape(bow_feats, shape=(-1, self._num_features, self._embedding_size)) )
        x_pooled = tf.reduce_sum(set_of_pooled, axis=-2)'''

        # simpler computation using residual masses
        rho = tf.expand_dims(P[:, 0], axis=-1)  # residual masses
        mixing_weights = (1. - self._num_features * rho) / mu

        if self._use_gemp:
            x_pooled = self.gmeanp(mixing_weights * inputs, training)
        else:
            x_pooled = self._global_pool(mixing_weights * bow_feats)

        # attribute embedding
        x_attr = tf.reduce_sum(ker_tpd, axis=[-2, -1])

        if training:
            x_emb = tf.concat([x_pooled, x_attr], axis=-1)
        else:
            x_emb = x_pooled
            if self._return_attention_maps:
                attention_maps = tf.transpose(P[:, 1:], perm=[0, 2, 3, 1]) * self._num_features

                out_maps = tf.keras.layers.Resizing(self._img_size[0], self._img_size[1])(attention_maps)

                x_emb = out_maps

        return x_emb

    @tf.function
    def fill_memory(self, embeddings):

        embeddings = self._partial_transport.maybe_normalize(tf.stop_gradient(embeddings))

        ptr_beg = self._mem_ptr
        ptr_end = self._mem_ptr + tf.shape(embeddings)[0]

        ptr_beg, ptr_end = tf.cond(
            pred=tf.greater(ptr_end, self._memory_size),
            true_fn=lambda: (self._memory_size - tf.shape(embeddings)[0], self._memory_size),
            false_fn=lambda: (ptr_beg, ptr_end)
        )

        self._memory[ptr_beg:ptr_end].assign(embeddings)

        self._mem_ptr.assign_add(tf.shape(embeddings)[0])

        self._ready.assign(tf.greater_equal(self._mem_ptr, self._memory_size))

        return self._ready

    @tf.function
    def initialize_support_kernel(self):

        initial_set = tf.expand_dims(self._memory[:self._support_size], axis=0)
        point_set = tf.expand_dims(self._memory[self._support_size:], axis=0)

        anchors = greedyKCenter(point_set, initial_set, self._support_size, normalized_embeddings=False)

        #tf.print('\033[3;34m' + '\nINFO:Model:Pooling: ' + 'Anchor features are initialized.' + '\033[0m')

        return self._partial_transport.kernel.assign(anchors)


    def call(self, inputs, training=True, **kwargs):


        if self._binary_costs == 'l2':
            if (self._memory_size > self._support_size) and (not self._return_attention_maps):
                emb = tf.cond(
                    pred=tf.equal(self._ready, tf.constant(True)),
                    true_fn=lambda: self.transport_pool(inputs, training),
                    false_fn=lambda: self.global_pool(inputs, training))
            else:
                emb = self.transport_pool(inputs, training)

        else: # convolution case
            emb = self.transport_pool(inputs, training)

        if self._augment_max_pool:
            x_max_pool = tf.keras.layers.GlobalMaxPooling2D(keepdims=True)(inputs)
            x_max_pool = self._pre_feature_head(x_max_pool)
            emb1 = tf.squeeze(x_max_pool) + emb[:, :self._embedding_size]
            if training:
                emb = tf.concat([emb1, emb[:, self._embedding_size:]], axis=-1)
            else:
                emb = emb1

        return emb

    def compute_output_shape(self, input_shape):
        out_dim = self._embedding_size
        output_shape = (input_shape[0], out_dim)

        if self._return_attention_maps:
            out_shape_2 = (input_shape[0],
                           self._img_size[0], self._img_size[1],
                           self._support_size)
            output_shape = out_shape_2#[output_shape, out_shape_2]

        return output_shape

    def get_config(self):
        config = super(GeneralizedSumPooling, self).get_config()
        config.update({'embedding_size': self._embedding_size,
                       'binary_costs': self._binary_costs,
                       'normalization': self._normalization,
                       'transport_ratio': self._transport_ratio,
                       'warm_up_steps': self._warm_up_steps,
                       'memory_size': self._memory_size,
                       'support_size': self._support_size,
                       'entropy_regularizer_weight': self._entropy_regularizer_weight,
                       'optimization_steps': self._optimization_steps,
                       'grad_method': self._grad_method,
                       'return_attention_maps': self._return_attention_maps,
                       'image_size': self._img_size,
                       'augment_max_pool': self._augment_max_pool,
                       'use_gemp': self._use_gemp
                       }
                      )
        return config

# meta global pooling
MetaGlobalPooling_cfg = CN()
MetaGlobalPooling_cfg.use_average = True
MetaGlobalPooling_cfg.use_max = False
MetaGlobalPooling_cfg.l2_normalize = False
MetaGlobalPooling_cfg.support_size = 32
MetaGlobalPooling_cfg.softmax_scale = 10.0
default.cfg.model.embedding_head.MetaGlobalPooling = MetaGlobalPooling_cfg

@register_pooling(name='MetaGlobalPooling')
@tf.keras.utils.register_keras_serializable()
class MetaGlobalPooling(tf.keras.layers.Layer):

    def __init__(self,
                 embedding_size,
                 support_size,
                 softmax_scale=10.,
                 l2_normalize=True,
                 use_average=True,
                 use_max=False,
                 name=None,
                 **kwargs):
        super(MetaGlobalPooling, self).__init__(name=name, **kwargs)

        self._embedding_size = embedding_size
        self._support_size = support_size
        self._softmax_scale = softmax_scale
        self._l2_normalize = l2_normalize
        self._use_average = use_average
        self._use_max = use_max

        self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self._max_pooling = tf.keras.layers.GlobalMaxPooling2D()

        self._transform = tf.keras.layers.Dense(units=embedding_size,
                                                use_bias=False,
                                                kernel_constraint=None,
                                                kernel_initializer='glorot_uniform',
                                                name='feature_transform')

        self._attr_transform = tf.keras.layers.Dense(units=support_size,
                                                use_bias=False,
                                                kernel_constraint=None,
                                                kernel_initializer='glorot_uniform',
                                                name='attr_transform')

        self._maybe_normalize = L2Normalization() if l2_normalize else \
            tf.keras.layers.Lambda(function=lambda x: x)

        self._normalize = L2Normalization()

    def call(self, inputs, training=True):

        x_feat = self._transform(inputs)

        x_emb = self._avg_pooling(x_feat)

        x_feat_normalized = self._normalize(x_feat)

        p_attr = tf.nn.softmax(self._softmax_scale * self._attr_transform(x_feat_normalized), axis=-1)
        x_attr = self._avg_pooling(p_attr)

        if training:
            return tf.concat([x_emb, x_attr], axis=-1)
        else:
            return x_emb



    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self._embedding_size)

        return output_shape

    def get_config(self):
        config = super(MetaGlobalPooling, self).get_config()
        config.update({'embedding_size': self._embedding_size,
                       'support_size': self._support_size,
                       'softmax_scale': self._softmax_scale,
                       'l2_normalize': self._l2_normalize,
                       'use_average': self._use_average,
                       'use_max': self._use_max
                       }
                      )
        return config
