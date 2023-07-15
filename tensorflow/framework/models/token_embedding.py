import tensorflow as tf
from ..layers import pooling as heads
from . import BaseModel
from ..layers.l2_normalization import L2Normalization
from ..layers.partial_transport import PartialTransport
from ..utilities.proxy_utils import greedyKCenter


class SelectivePooling(tf.keras.layers.Layer):

    def __init__(self, bag_size, cfg, name=None, **kwargs):

        super(SelectivePooling, self).__init__(name=name, **kwargs)

        self.partial_transport = PartialTransport(
            support_size=cfg.model.embedding_head.MetaTransportPooling.support_size,
            cost_fn='l2',
            normalization='l2',
            entropy_regularizer_weight=cfg.model.embedding_head.MetaTransportPooling.entropy_regularizer_weight,
            optimization_steps=cfg.model.embedding_head.MetaTransportPooling.optimization_steps,
            grad_method='inv',
            name='partial_transport')

        self.zeros_cost = tf.keras.models.Sequential(
            [
                tf.keras.layers.Lambda(
                    lambda x: tf.stop_gradient(x)),
                tf.keras.layers.Lambda(
                    lambda x: tf.reduce_sum(0.0 * x, axis=-1))
            ],
            name='zeros_cost'
        )

        mu = cfg.model.embedding_head.MetaTransportPooling.transport_ratio
        self.constant_ratio = tf.keras.models.Sequential(
            [
                tf.keras.layers.Lambda(lambda x: tf.stop_gradient(x)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Lambda(lambda x:
                                       tf.reduce_sum(0.0 * x, axis=-1) + mu),
            ],
            name='constant_ratio'
        )

        self.bag_size = bag_size

        self._support_size = cfg.model.embedding_head.MetaTransportPooling.support_size

        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D()

        self._steps = self.add_weight(shape=(),
                                      dtype=tf.int32,
                                      trainable=False,
                                      initializer=tf.keras.initializers.zeros(),
                                      constraint=None,
                                      name='{}/step_counter'.format(self.name))

        self._warm_up_steps = 4096

        self._memory_size = 1024

        self._feat_emb_size = cfg.model.embedding_head.embedding_size

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

    @tf.function
    def global_pool(self, bow_feats, training):

        x_emb = self.global_pooling(bow_feats)

        if training:
            self._steps.assign_add(1)

        is_memory_ready = tf.cond(
            pred=tf.greater(self._steps, tf.constant(self._warm_up_steps)),
            true_fn=lambda: self.fill_memory(x_emb),
            false_fn=lambda: self._ready)

        tf.cond(
            pred=tf.equal(is_memory_ready, tf.constant(True)),
            true_fn=self.initialize_support_kernel,
            false_fn=lambda: self.partial_transport.kernel
        )

        emb = x_emb

        return emb

    @tf.function
    def transport_pool(self, bow_feats, training):
        unary_costs = self.zeros_cost(bow_feats)
        transport_ratio = self.constant_ratio(bow_feats)

        P = self.partial_transport([bow_feats, transport_ratio, unary_costs])

        mu = tf.stop_gradient(transport_ratio)
        num_expands = len(P.get_shape().as_list()) - 1
        for _ in range(num_expands):
            mu = tf.expand_dims(mu, axis=-1)

        rho = tf.expand_dims(P[:, 0], axis=-1)  # residual masses
        mixing_weights = (1. - self.bag_size * rho) / mu
        emb = self.global_pooling(mixing_weights * bow_feats)

        return emb

    @tf.function
    def fill_memory(self, embeddings):

        embeddings = L2Normalization()(tf.stop_gradient(embeddings))

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

        tf.print('\033[3;34m' + '\nINFO:Model:Pooling: ' + 'Anchor features are initialized.' + '\033[0m')

        return self.partial_transport.kernel.assign(anchors)

    def call(self, inputs, training=True, **kwargs):
        x_bow = inputs



        emb = tf.cond(
            pred=tf.equal(self._ready, tf.constant(True)),
            true_fn=lambda: self.transport_pool(x_bow, training),
            false_fn=lambda: self.global_pool(x_bow, training))


        return emb


def Word2Vec(bag_size, vocab_size, cfg):

    # input layer
    inputs = tf.keras.Input(shape=(bag_size, vocab_size), dtype=tf.float32, name='Input')

    reshape = tf.keras.layers.Reshape(target_shape=(bag_size, 1, vocab_size))

    normalize = L2Normalization()

    limits = 0.3
    kernel_constraint = lambda w: tf.clip_by_value(w, -limits, limits)
    embedding_lookup = tf.keras.layers.Dense(units=cfg.model.embedding_head.embedding_size,
                                             use_bias=False,
                                             kernel_constraint=kernel_constraint,
                                             kernel_initializer='glorot_uniform',
                                             name='feature_transform')

    if cfg.model.embedding_head.arch == 'GlobalPooling':
        pooling = tf.keras.layers.GlobalAveragePooling2D(name='EmbeddingHead')
    else:
        pooling = SelectivePooling(bag_size, cfg, name='EmbeddingHead')
        pooling.partial_transport._kernel_constraint = kernel_constraint


    # constructing the model
    bow = reshape(inputs)
    feats = embedding_lookup(bow)
    #n_feats = normalize(feats)
    emb = pooling(feats)

    name = 'Pseudo-Word2Vec'

    model = BaseModel(inputs=inputs, outputs=emb, name=name)

    return model
