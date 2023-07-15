import tensorflow as tf
from ..l2_normalization import L2Normalization

from ...configs import default
from ...configs.config import CfgNode as CN

from . import register_pooling

# global pooling
DeLF_cfg = CN()
DeLF_cfg.num_filters = 512
DeLF_cfg.conv_size = 1
DeLF_cfg.l2_normalize = False

default.cfg.model.embedding_head.DeLF = DeLF_cfg

@register_pooling(name='DeLF')
@tf.keras.utils.register_keras_serializable()
class DeLF(tf.keras.layers.Layer):
    '''@inproceedings{noh2017large,
      title={Large-scale image retrieval with attentive deep local features},
      author={Noh, Hyeonwoo and Araujo, Andre and Sim, Jack and Weyand, Tobias and Han, Bohyung},
      booktitle={Proceedings of the IEEE international conference on computer vision},
      pages={3456--3465},
      year={2017}
    }'''

    def __init__(self,
                 embedding_size,
                 l2_normalize=True,
                 num_filters=512,
                 conv_size=1,
                 name=None,
                 **kwargs):
        super(DeLF, self).__init__(name=name, **kwargs)

        self._embedding_size = embedding_size
        self._l2_normalize = l2_normalize
        self._num_filters = num_filters
        self._conv_size = conv_size

        self._gap = tf.keras.layers.GlobalAveragePooling2D()

        self._transform = tf.keras.layers.Dense(units=self._embedding_size,
                                                use_bias=False,
                                                kernel_constraint=None,
                                                kernel_initializer='glorot_uniform',
                                                name='feature_transform')

        self._att_head = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(filters=self._num_filters,
                                       kernel_size=self._conv_size,
                                       padding='same',
                                       strides=1,
                                       activation='relu',
                                       name='attention_conv_1'),
                tf.keras.layers.Dense(units=1,
                                      activation='softplus',
                                      name='attention_conv_2')
            ],
            name='attention_head'
        )

        self._maybe_normalize = L2Normalization() if self._l2_normalize else \
            tf.keras.layers.Lambda(function=lambda x: x)


    def call(self, inputs, **kwargs):

        weigths = self._att_head(inputs)

        masked = tf.multiply(weigths, inputs)

        x_pooled = self._gap(masked)

        x_emb = self._transform(x_pooled)

        x_emb = self._maybe_normalize(x_emb)

        return x_emb


    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self._embedding_size)

        return output_shape

    def get_config(self):
        config = super(DeLF, self).get_config()
        config.update({'embedding_size': self._embedding_size,
                       'l2_normalize': self._l2_normalize,
                       'num_filters': self._num_filters,
                       'conv_size': self._conv_size
                       }
                      )
        return config



