import tensorflow as tf
from ..l2_normalization import L2Normalization

from ...configs import default
from ...configs.config import CfgNode as CN

from . import register_pooling

# global pooling
CBAM_cfg = CN()
CBAM_cfg.channel_reduction = 16
CBAM_cfg.conv_size = 3
CBAM_cfg.l2_normalize = False

default.cfg.model.embedding_head.CBAM = CBAM_cfg

@register_pooling(name='CBAM')
@tf.keras.utils.register_keras_serializable()
class CBAM(tf.keras.layers.Layer):
    '''@inproceedings{woo2018cbam,
      title={Cbam: Convolutional block attention module},
      author={Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and Kweon, In So},
      booktitle={Proceedings of the European conference on computer vision (ECCV)},
      pages={3--19},
      year={2018}
    }'''

    def __init__(self,
                 embedding_size,
                 l2_normalize=True,
                 channel_reduction=16,
                 conv_size=3,
                 name=None,
                 **kwargs):
        super(CBAM, self).__init__(name=name, **kwargs)

        self._embedding_size = embedding_size
        self._l2_normalize = l2_normalize
        self._channel_reduction = channel_reduction
        self._conv_size = conv_size



    def build(self, input_shape):

        d = input_shape[-1]

        self._avg_pooling_2d = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self._max_pooling_2d = tf.keras.layers.GlobalMaxPooling2D(keepdims=True)

        self._gap = tf.keras.layers.GlobalAveragePooling2D()

        self._avg_pooling_1d = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))
        self._max_pooling_1d = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))

        self._transform = tf.keras.layers.Dense(units=self._embedding_size,
                                                use_bias=False,
                                                kernel_constraint=None,
                                                kernel_initializer='glorot_uniform',
                                                name='feature_transform')

        self._chatt_head = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(units=d // self._channel_reduction,
                                      use_bias=False,
                                      activation='relu',
                                      name='channel_MLP_1'),
                tf.keras.layers.Dense(units=d,
                                      use_bias=False,
                                      name='channel_MLP_2')
            ],
            name='channel_attention_MLP'
        )

        self._spatt_head = tf.keras.layers.Conv2D(filters=1,
                                                  kernel_size=self._conv_size,
                                                  padding='same',
                                                  use_bias=False,
                                                  activation='sigmoid',
                                                  name='spatial_conv'
                                                  )

        self._maybe_normalize = L2Normalization() if self._l2_normalize else \
            tf.keras.layers.Lambda(function=lambda x: x)




    def call(self, inputs, **kwargs):

        features = inputs # self._transform(inputs)

        # channel attention

        # max path
        channel_max = self._max_pooling_2d(features)
        max_att = self._chatt_head(channel_max)

        # mean path
        channnel_avg = self._avg_pooling_2d(features)
        avg_att = self._chatt_head(channnel_avg)

        channel_att = tf.nn.sigmoid(max_att + avg_att)

        features = tf.multiply(features, channel_att)

        # spatial attention

        spatial_max = self._max_pooling_1d(features)
        spatial_avg = self._avg_pooling_1d(features)

        spatial_att = self._spatt_head(tf.concat([spatial_max, spatial_avg], axis=-1))

        features = tf.multiply(features, spatial_att)

        x_pooled = self._gap(features)

        x_emb = self._transform(x_pooled)

        x_emb = self._maybe_normalize(x_emb)

        return x_emb


    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self._embedding_size)

        return output_shape

    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({'embedding_size': self._embedding_size,
                       'l2_normalize': self._l2_normalize,
                       'channel_reduction': self._channel_reduction,
                       'conv_size': self._conv_size
                       }
                      )
        return config



