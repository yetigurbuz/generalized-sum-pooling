import tensorflow as tf
from ..l2_normalization import L2Normalization

from ...configs import default
from ...configs.config import CfgNode as CN

from . import register_pooling

# global pooling
CROW_cfg = CN()
CROW_cfg.l2_normalize = False

default.cfg.model.embedding_head.CroW = CROW_cfg

@register_pooling(name='CroW')
@tf.keras.utils.register_keras_serializable()
class CroW(tf.keras.layers.Layer):
    '''@inproceedings{kalantidis2016cross,
      title={Cross-dimensional weighting for aggregated deep convolutional features},
      author={Kalantidis, Yannis and Mellina, Clayton and Osindero, Simon},
      booktitle={European conference on computer vision},
      pages={685--701},
      year={2016},
      organization={Springer}
        }'''

    def __init__(self,
                 embedding_size,
                 l2_normalize=True,
                 name=None,
                 **kwargs):
        super(CroW, self).__init__(name=name, **kwargs)

        self._embedding_size = embedding_size
        self._l2_normalize = l2_normalize

        self._gap = tf.keras.layers.GlobalAveragePooling2D()
        self._avg_pooling_2d = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)


        self._transform = tf.keras.layers.Dense(units=embedding_size,
                                                use_bias=False,
                                                kernel_constraint=None,
                                                kernel_initializer='glorot_uniform',
                                                name='feature_transform')



        self._maybe_normalize = L2Normalization() if l2_normalize else \
            tf.keras.layers.Lambda(function=lambda x: x)

    def call(self, inputs, **kwargs):

        features = tf.nn.relu(inputs) #self._transform(inputs)

        # spatial attention

        s_prime = tf.reduce_sum(features, axis=-1, keepdims=True)
        s_norm = tf.sqrt(
            tf.reduce_sum(
                tf.square(s_prime),
                axis=[1, 2], keepdims=True)
        )
        spatt = tf.sqrt(tf.divide(s_prime, s_norm))


        # channel attention

        q = self._avg_pooling_2d(tf.nn.sigmoid(features))
        K = tf.cast(tf.shape(features)[-1], tf.float32)
        eps = 1.0e-16

        chatt = tf.math.log(
            tf.divide(K * eps + tf.reduce_sum(q, axis=-1, keepdims=True),
                      eps + q)

        )

        features = tf.multiply(features,
                               tf.multiply(spatt, chatt))

        x_pooled = self._gap(features)

        x_emb = self._transform(x_pooled)

        x_emb = self._maybe_normalize(x_emb)

        return x_emb


    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self._embedding_size)

        return output_shape

    def get_config(self):
        config = super(CroW, self).get_config()
        config.update({'embedding_size': self._embedding_size,
                       'l2_normalize': self._l2_normalize
                       }
                      )
        return config



