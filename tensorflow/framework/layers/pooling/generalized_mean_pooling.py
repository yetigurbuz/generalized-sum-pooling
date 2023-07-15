import tensorflow as tf
from ..l2_normalization import L2Normalization

from ...configs import default
from ...configs.config import CfgNode as CN

from . import register_pooling

# global pooling
GMeanP_cfg = CN()
GMeanP_cfg.power = 3.0
GMeanP_cfg.learn_power = True
GMeanP_cfg.shared_power = True
GMeanP_cfg.l2_normalize = False

default.cfg.model.embedding_head.GMeanP = GMeanP_cfg

@register_pooling(name='GMeanP')
@tf.keras.utils.register_keras_serializable()
class GMeanP(tf.keras.layers.Layer):
    '''@article{radenovic2018fine,
          title={Fine-tuning CNN image retrieval with no human annotation},
          author={Radenovi{\'c}, Filip and Tolias, Giorgos and Chum, Ond{\v{r}}ej},
          journal={IEEE transactions on pattern analysis and machine intelligence},
          volume={41},
          number={7},
          pages={1655--1668},
          year={2018},
          publisher={IEEE}
        }'''

    def __init__(self,
                 embedding_size,
                 l2_normalize=False,
                 power=3.0,
                 learn_power=True,
                 shared_power=True,
                 name=None,
                 **kwargs):
        super(GMeanP, self).__init__(name=name, **kwargs)

        self._embedding_size = embedding_size
        self._l2_normalize = l2_normalize
        self._power = power
        self._learn_power = learn_power
        self._shared_power = shared_power



    def build(self, input_shape):

        b, w, h, d = input_shape

        if self._shared_power:
            power_shape = (1, 1, 1, 1)
        else:
            power_shape = (1, 1, 1, d)

        self._p = self.add_weight(
            name="power", shape=power_shape,
            initializer=tf.keras.initializers.constant(
                tf.ones(shape=power_shape) * self._power),
            trainable=self._learn_power)


        self._transform = tf.keras.layers.Dense(units=self._embedding_size,
                                                use_bias=False,
                                                kernel_constraint=None,
                                                kernel_initializer='glorot_uniform',
                                                name='feature_transform')

        self._gap = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)

        self._maybe_normalize = L2Normalization() if self._l2_normalize else \
            tf.keras.layers.Lambda(function=lambda x: x)

        self._reshape = tf.keras.layers.Reshape(target_shape=[self._embedding_size])


    def call(self, inputs, **kwargs):

        x = inputs #self._transform(inputs) # shape: b, w, h, d

        x = tf.math.maximum(x, 1e-6)

        x_to_p = tf.pow(x, self._p)

        x_to_p_avg = self._gap(x_to_p)

        x_pooled = tf.pow(x_to_p_avg, 1.0 / self._p)

        x_emb = self._reshape(self._transform(x_pooled))

        x_emb = self._maybe_normalize(x_emb)

        return x_emb


    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self._embedding_size)

        return output_shape

    def get_config(self):
        config = super(GMeanP, self).get_config()
        config.update({'embedding_size': self._embedding_size,
                       'l2_normalize': self._l2_normalize,
                       'power': self._power,
                       'learn_power': self._learn_power,
                       'shared_power': self._shared_power
                       }
                      )
        return config
