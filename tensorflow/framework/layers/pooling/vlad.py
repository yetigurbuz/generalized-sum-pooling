import tensorflow as tf
from ..l2_normalization import L2Normalization

from ...configs import default
from ...configs.config import CfgNode as CN

from . import register_pooling

# global pooling
VLAD_cfg = CN()
VLAD_cfg.num_centers = 16
VLAD_cfg.l2_normalize = False

default.cfg.model.embedding_head.VLAD = VLAD_cfg

@register_pooling(name='VLAD')
@tf.keras.utils.register_keras_serializable()
class VLAD(tf.keras.layers.Layer):
    '''@inproceedings{arandjelovic2016netvlad,
          title={NetVLAD: CNN architecture for weakly supervised place recognition},
          author={Arandjelovic, Relja and Gronat, Petr and Torii, Akihiko and Pajdla, Tomas and Sivic, Josef},
          booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
          pages={5297--5307},
          year={2016}
}'''

    def __init__(self,
                 embedding_size,
                 l2_normalize=True,
                 num_centers=64,
                 name=None,
                 **kwargs):
        super(VLAD, self).__init__(name=name, **kwargs)

        self._embedding_size = embedding_size
        self._feat_size = embedding_size // num_centers
        self._l2_normalize = l2_normalize
        self._num_centers = num_centers

        self._reshape = tf.keras.layers.Reshape(target_shape=[embedding_size])

        # pre feature transform
        self._transform = tf.keras.layers.Dense(units=self._feat_size,
                                                use_bias=False,
                                                kernel_constraint=None,
                                                kernel_initializer='glorot_uniform',
                                                name='feature_transform')

        # centers
        self._centers = self.add_weight(shape=(1, 1, 1, self._num_centers, self._feat_size),
                                        initializer=tf.keras.initializers.GlorotUniform(),
                                        name='{}/centers'.format(self.name))

        # center assignment weights
        self._assignment_weights = tf.keras.layers.Dense(units=self._num_centers,
                                                         use_bias=True,
                                                         activation='softmax',
                                                         name='assignment_weights')

        self._maybe_normalize = L2Normalization() if self._l2_normalize else \
            tf.keras.layers.Lambda(function=lambda x: x)

    def call(self, inputs, **kwargs):

        features = self._transform(inputs)

        # assignment weights (shape = (b, w, h, c) )
        a = self._assignment_weights(features)
        a = tf.expand_dims(a, axis=-1) # shape = (b, w, h, c, 1)

        # residuals
        r = tf.subtract(tf.expand_dims(features, axis=-2), # shape = (b, w, h, 1, d)
                        self._centers) # shape = (1, 1, 1, c, d)

        # weighted sum of residuals
        v = tf.reduce_sum(tf.multiply(a, r), axis=(1, 2)) # shape = (b, c, d)

        # intra normalized
        v_normed = tf.nn.l2_normalize(v, axis=-1)

        # concatenated feature
        x_vlad = self._reshape(v_normed)

        x_emb = self._maybe_normalize(x_vlad)

        return x_emb


    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self._embedding_size)

        return output_shape

    def get_config(self):
        config = super(VLAD, self).get_config()
        config.update({'embedding_size': self._embedding_size,
                       'l2_normalize': self._l2_normalize,
                       'num_centers': self._num_centers
                       }
                      )
        return config



