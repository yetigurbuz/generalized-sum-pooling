import tensorflow as tf
from ..l2_normalization import L2Normalization

from ...configs import default
from ...configs.config import CfgNode as CN

from ..partial_transport import MovingAverageStandardization as BatchNormalization
from ..pooling.generalized_mean_pooling import GMeanP

from . import register_pooling

# global pooling
SOAP_cfg = CN()
SOAP_cfg.channel_reduction = 1
SOAP_cfg.l2_normalize = False

default.cfg.model.embedding_head.SOAP = SOAP_cfg

@register_pooling(name='SOAP')
@tf.keras.utils.register_keras_serializable()
class SOAP(tf.keras.layers.Layer):
    ''' @inproceedings{ng2020solar,
            author    = {Ng, Tony and Balntas, Vassileios and Tian, Yurun and Mikolajczyk, Krystian},
            title     = {{SOLAR}: Second-Order Loss and Attention for Image Retrieval},
            booktitle = {ECCV},
            year      = {2020}
        }'''

    def __init__(self,
                 embedding_size,
                 l2_normalize=True,
                 channel_reduction=1,
                 name=None,
                 **kwargs):
        super(SOAP, self).__init__(name=name, **kwargs)

        self._embedding_size = embedding_size
        self._l2_normalize = l2_normalize
        self._channel_reduction = channel_reduction


    def build(self, input_shape):

        b, h, w, d = input_shape

        self._att_dim = d // self._channel_reduction

        # query head
        self._query_head = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(filters=self._att_dim,
                                       kernel_size=1, strides=1, padding='same'),
                BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Reshape(target_shape=(h * w, self._att_dim))
            ]

        )

        # key head
        self._key_head = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(filters=self._att_dim,
                                       kernel_size=1, strides=1, padding='same'),
                BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Reshape(target_shape=(h * w, self._att_dim))
            ]

        )

        # value head
        self._value_head =  tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(filters=self._att_dim,
                                    kernel_size=1, strides=1, padding='same'),
                tf.keras.layers.Reshape(target_shape=(h * w, self._att_dim))
            ]
        )

        # transform
        self._transform = tf.keras.models.Sequential(
            [
                tf.keras.layers.Reshape(target_shape=(h, w, self._att_dim)),
                tf.keras.layers.Conv2D(filters=d,
                                       kernel_size=1, strides=1, padding='same')
            ]
        )

        self._sm = tf.keras.layers.Softmax(axis=-1)

        self._gem = GMeanP(embedding_size=self._embedding_size, l2_normalize=self._l2_normalize)

    def call(self, inputs, **kwargs):

        features = inputs

        # query
        q = self._query_head(features) # shape: (b, wh, d)

        # key
        k = self._key_head(features) # shape: (b, wh, d)

        # value
        v = self._value_head(features) # shape: (b, wh, d)

        z = tf.matmul(q, k, transpose_b=True) # shape: (b, wh, wh) => z_ij = q_i T k_j

        t = tf.pow(tf.cast(self._att_dim, tf.float32), -.5)
        a = self._sm(t * z)

        z = tf.matmul(a, v) # shape: (b, wh, d)

        z = self._transform(z) # shape: (b, w, h, d)

        x_enhanced = features + z

        x_pooled = self._gem(x_enhanced)

        return x_pooled


    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self._embedding_size)

        return output_shape

    def get_config(self):
        config = super(SOAP, self).get_config()
        config.update({'embedding_size': self._embedding_size,
                       'l2_normalize': self._l2_normalize,
                       'channel_reduction': self._channel_reduction
                       }
                      )
        return config



