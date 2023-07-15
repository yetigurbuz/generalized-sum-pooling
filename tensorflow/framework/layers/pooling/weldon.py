import tensorflow as tf
from ..l2_normalization import L2Normalization

from ...configs import default
from ...configs.config import CfgNode as CN

from . import register_pooling

# global pooling
WELDON_cfg = CN()
WELDON_cfg.k_max = 3
WELDON_cfg.k_min = 3
WELDON_cfg.l2_normalize = False

default.cfg.model.embedding_head.WELDON = WELDON_cfg

@register_pooling(name='WELDON')
@tf.keras.utils.register_keras_serializable()
class WELDON(tf.keras.layers.Layer):
    '''@INPROCEEDINGS{7780882,
          author={Durand, Thibaut and Thome, Nicolas and Cord, Matthieu},
          booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
          title={WELDON: Weakly Supervised Learning of Deep Convolutional Neural Networks},
          year={2016},
          volume={},
          number={},
          pages={4743-4752},
          doi={10.1109/CVPR.2016.513}}'''

    def __init__(self,
                 embedding_size,
                 l2_normalize=True,
                 k_max=3,
                 k_min=3,
                 name=None,
                 **kwargs):
        super(WELDON, self).__init__(name=name, **kwargs)

        self._embedding_size = embedding_size
        self._l2_normalize = l2_normalize
        self._k_max = k_max
        self._k_min = k_min

        self._transform = tf.keras.layers.Dense(units=self._embedding_size,
                                                use_bias=False,
                                                kernel_constraint=None,
                                                kernel_initializer='glorot_uniform',
                                                name='feature_transform')

        self._maybe_normalize = L2Normalization() if self._l2_normalize else \
            tf.keras.layers.Lambda(function=lambda x: x)


    def pool_kmax(self, features):

        dims = tf.shape(features)

        num_feats = dims[1] * dims[2]

        features_flattened = tf.reshape(
            tf.transpose(features, perm=[0, 3, 1, 2]),
            shape=(dims[0], dims[3], num_feats)
        )

        # get the value of k-largest activation for each channel
        k_largest = tf.math.top_k(features_flattened,
                                  k=self._k_max,
                                  sorted=True)[0][:, :, -1]

        max_mask = tf.cast(
            tf.greater_equal(features_flattened,
                             tf.expand_dims(k_largest, axis=-1)), tf.float32)

        k_max_pooled = tf.reduce_sum(
            tf.multiply(features_flattened, max_mask),
            axis=-1)

        return k_max_pooled


    def pool_kmin(self, features):

        dims = tf.shape(features)

        num_feats = dims[1] * dims[2]

        features_flattened = tf.reshape(
            tf.transpose(features, perm=[0, 3, 1, 2]),
            shape=(dims[0], dims[3], num_feats)
        )

        # get the value of k-smallest activation for each channel
        k_smallest = tf.math.top_k(-features_flattened,
                                  k=self._k_min,
                                  sorted=True)[0][:, :, -1]

        min_mask = tf.cast(
            tf.greater_equal(-features_flattened,
                             tf.expand_dims(k_smallest, axis=-1)), tf.float32)

        k_min_pooled = tf.reduce_sum(
            tf.multiply(features_flattened, min_mask),
            axis=-1)

        return k_min_pooled




    def call(self, inputs, **kwargs):

        features = self._transform(inputs)


        k_max_pooled = self.pool_kmax(features)

        k_min_pooled = self.pool_kmin(features)

        x_pooled = k_max_pooled + k_min_pooled

        x_emb = self._maybe_normalize(x_pooled)

        return x_emb


    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self._embedding_size)

        return output_shape

    def get_config(self):
        config = super(WELDON, self).get_config()
        config.update({'embedding_size': self._embedding_size,
                       'l2_normalize': self._l2_normalize,
                       'k_max': self._k_max,
                       'k_min': self._k_min
                       }
                      )
        return config



