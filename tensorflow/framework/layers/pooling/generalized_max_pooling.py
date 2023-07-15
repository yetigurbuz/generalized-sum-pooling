import tensorflow as tf
from ..l2_normalization import L2Normalization

from ...configs import default
from ...configs.config import CfgNode as CN

from . import register_pooling

# global pooling
GMaxP_cfg = CN()
GMaxP_cfg.regression_lambda = 1.0
GMaxP_cfg.l2_normalize = False

default.cfg.model.embedding_head.GMaxP = GMaxP_cfg

@register_pooling(name='GMaxP')
@tf.keras.utils.register_keras_serializable()
class GMaxP(tf.keras.layers.Layer):
    '''@article{murray2016interferences,
      title={Interferences in match kernels},
      author={Murray, Naila and J{\'e}gou, Herv{\'e} and Perronnin, Florent and Zisserman, Andrew},
      journal={IEEE transactions on pattern analysis and machine intelligence},
      volume={39},
      number={9},
      pages={1797--1810},
      year={2016},
      publisher={IEEE}
    }'''

    def __init__(self,
                 embedding_size,
                 l2_normalize=True,
                 regression_lambda=1.0,
                 name=None,
                 **kwargs):
        super(GMaxP, self).__init__(name=name, **kwargs)

        self._embedding_size = embedding_size
        self._l2_normalize = l2_normalize
        self._regression_lambda = regression_lambda



    def build(self, input_shape):

        b, w, h, d = input_shape

        self._flatten = tf.keras.layers.Reshape(target_shape=(w * h, d))#self._embedding_size))




        self._transform = tf.keras.layers.Dense(units=self._embedding_size,
                                                use_bias=False,
                                                kernel_constraint=None,
                                                kernel_initializer='glorot_uniform',
                                                name='feature_transform')

        self._maybe_normalize = L2Normalization() if self._l2_normalize else \
            tf.keras.layers.Lambda(function=lambda x: x)



    def call(self, inputs, **kwargs):

        x = inputs #self._transform(inputs) # shape: b, w, h, d

        p = self._flatten(x) # shape: b, wh, d

        ppT = tf.matmul(p, p, transpose_b=True) # shape: b, wh, wh



        num_samples = tf.shape(p)[1]

        reg = self._regression_lambda * tf.eye(num_samples, batch_shape=[1])
        inv_ppT_plus_reg = tf.linalg.inv(ppT + reg)
        # optimal pooling pT inv(ppT + lambda I) 1
        x_pooled = tf.reduce_sum(
            tf.matmul(p, inv_ppT_plus_reg, transpose_a=True),
            axis=-1)

        x_pooled = self._transform(x_pooled)
        x_emb = self._maybe_normalize(x_pooled)
        return x_emb


    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self._embedding_size)

        return output_shape

    def get_config(self):
        config = super(GMaxP, self).get_config()
        config.update({'embedding_size': self._embedding_size,
                       'l2_normalize': self._l2_normalize,
                       'regression_lambda': self._regression_lambda
                       }
                      )
        return config




