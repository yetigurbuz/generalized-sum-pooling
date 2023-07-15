import tensorflow as tf

from ..optimal_transport import OptimalTransport
from ..l2_normalization import L2Normalization

from ...utilities.proxy_utils import greedyKCenter

from ...configs import default
from ...configs.config import CfgNode as CN

from . import register_pooling

# generalized sum pooling
OTP_cfg = CN()
OTP_cfg.binary_costs = 'l2'
OTP_cfg.normalization = 'l2'
OTP_cfg.support_size = 64
OTP_cfg.warm_up_steps = 0
OTP_cfg.memory_size = 1024
OTP_cfg.entropy_regularizer_weight = 2.0
OTP_cfg.optimization_steps = 10

default.cfg.model.embedding_head.OTP = OTP_cfg

@register_pooling(name='OTP')
@tf.keras.utils.register_keras_serializable()
class OTP(tf.keras.layers.Layer):

    def __init__(self,
                 embedding_size,
                 support_size,
                 binary_costs='l2',
                 normalization='l2',  # or ema std or None
                 warm_up_steps=0,  # steps before start prediction of constraints
                 memory_size=1024,
                 entropy_regularizer_weight=2.,
                 optimization_steps=10,
                 name=None,
                 **kwargs):

        super(OTP, self).__init__(name=name, **kwargs)

        self._embedding_size = embedding_size
        self._normalization = normalization
        self._binary_costs = binary_costs.lower()
        self._support_size = support_size
        self._entropy_regularizer_weight = entropy_regularizer_weight
        self._optimization_steps = optimization_steps


        self._steps = self.add_weight(shape=(),
                                      dtype=tf.int32,
                                      trainable=False,
                                      initializer=tf.keras.initializers.zeros(),
                                      constraint=None,
                                      name='{}/step_counter'.format(self.name))
        self._warm_up_steps = warm_up_steps

        self._memory_size = memory_size

        self._feat_emb_size = self._embedding_size // self._support_size

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


    def build(self, input_shape):

        w, h = input_shape[1:3]  # 2D to 1D bag of features representation
        self._num_features = w * h

        self._flatten = tf.keras.layers.Reshape(target_shape=(self._num_features, self._feat_emb_size))

        self._concat_feats = tf.keras.layers.Reshape(target_shape=[self._embedding_size])

        # feature transform heads
        self._pre_feature_head = tf.keras.layers.Conv2D(filters=self._feat_emb_size,
                                                        kernel_size=1,
                                                        padding='SAME',
                                                        use_bias=False,
                                                        kernel_initializer='glorot_uniform',
                                                        name='feature_transform')


        # pooling heads
        self._optimal_transport = OptimalTransport(
            support_size=self._support_size,
            cost_fn=self._binary_costs,
            normalization=self._normalization,
            entropy_regularizer_weight=self._entropy_regularizer_weight,
            optimization_steps=self._optimization_steps)

        self._global_pool = tf.keras.layers.GlobalAveragePooling2D()

    @tf.function
    def global_pool(self, bow_feats, training):

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
            false_fn=lambda: self._optimal_transport.kernel
        )

        zeros = tf.zeros(shape=(tf.shape(x_emb)[0],
                                self._embedding_size - self._feat_emb_size))

        emb = tf.concat([x_emb, zeros], axis=-1)

        return emb

    @tf.function
    def transport_pool(self, bow_feats, training):

        # transport kernel
        ker_tpd = self._optimal_transport(bow_feats)


        # ensemble way of computation of pooled features
        set_of_pooled = ker_tpd @ self._flatten(bow_feats)

        x_emb = tf.multiply(tf.sqrt(tf.cast(self._support_size, tf.float32)),
                            self._concat_feats(set_of_pooled))


        return x_emb

    @tf.function
    def fill_memory(self, embeddings):

        embeddings = self._optimal_transport.maybe_normalize(tf.stop_gradient(embeddings))

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

        return self._optimal_transport.kernel.assign(anchors)


    def call(self, inputs, training=True, **kwargs):

        x_bow = self._pre_feature_head(inputs)

        if self._binary_costs == 'l2':
            if (self._memory_size > self._support_size):
                emb = tf.cond(
                    pred=tf.equal(self._ready, tf.constant(True)),
                    true_fn=lambda: self.transport_pool(x_bow, training),
                    false_fn=lambda: self.global_pool(x_bow, training))
            else:
                emb = self.transport_pool(x_bow, training)

        else: # convolution case
            emb = self.transport_pool(x_bow, training)

        return emb

    def compute_output_shape(self, input_shape):
        out_dim = self._embedding_size
        output_shape = (input_shape[0], out_dim)

        return output_shape

    def get_config(self):
        config = super(OTP, self).get_config()
        config.update({'embedding_size': self._embedding_size,
                       'binary_costs': self._binary_costs,
                       'normalization': self._normalization,
                       'warm_up_steps': self._warm_up_steps,
                       'memory_size': self._memory_size,
                       'support_size': self._support_size,
                       'entropy_regularizer_weight': self._entropy_regularizer_weight,
                       'optimization_steps': self._optimization_steps
                       }
                      )
        return config
