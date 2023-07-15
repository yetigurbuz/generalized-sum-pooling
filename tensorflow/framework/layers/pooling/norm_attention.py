import tensorflow as tf
from ..l2_normalization import L2Normalization

from ...configs import default
from ...configs.config import CfgNode as CN

from . import register_pooling

# global pooling
NAP_cfg = CN()
NAP_cfg.pool_size = 3
NAP_cfg.l2_normalize = False

default.cfg.model.embedding_head.NAP = NAP_cfg

@register_pooling(name='NAP')
@tf.keras.utils.register_keras_serializable()
class NAP(tf.keras.layers.Layer):
    '''@inproceedings{tolias2020learning,
      title={Learning and aggregating deep local descriptors for instance-level recognition},
      author={Tolias, Giorgos and Jenicek, Tomas and Chum, Ond{\v{r}}ej},
      booktitle={European Conference on Computer Vision},
      pages={460--477},
      year={2020},
      organization={Springer}
    }'''

    def __init__(self,
                 embedding_size,
                 l2_normalize=True,
                 pool_size=3,
                 name=None,
                 **kwargs):
        super(NAP, self).__init__(name=name, **kwargs)

        self._embedding_size = embedding_size
        self._l2_normalize = l2_normalize
        self._pool_size = pool_size

        self._smooth = tf.keras.layers.AveragePooling2D(pool_size=self._pool_size,
                                                        strides=1,
                                                        padding='same')

        self._transform = tf.keras.layers.Dense(units=self._embedding_size,
                                                use_bias=True,
                                                kernel_constraint=None,
                                                kernel_initializer='glorot_uniform',
                                                name='feature_transform')

        self._maybe_normalize = L2Normalization() if self._l2_normalize else \
            tf.keras.layers.Lambda(function=lambda x: x)

        self._gap = tf.keras.layers.GlobalAveragePooling2D()




    def call(self, inputs, **kwargs):

        features = inputs

        # spatial attention

        feat_norm = tf.sqrt(
            tf.reduce_sum(
                tf.square(features),
                axis=-1, keepdims=True)
        )

        attention = tf.stop_gradient(feat_norm)

        # feature transform path

        smoothed = self._smooth(features)

        transformed = self._transform(smoothed)

        weighted_features = tf.multiply(transformed, attention)

        x_pooled = self._gap(weighted_features)

        x_emb = self._maybe_normalize(x_pooled)

        return x_emb


    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self._embedding_size)

        return output_shape

    def get_config(self):
        config = super(NAP, self).get_config()
        config.update({'embedding_size': self._embedding_size,
                       'l2_normalize': self._l2_normalize,
                       'pool_size': self._pool_size
                       }
                      )
        return config



