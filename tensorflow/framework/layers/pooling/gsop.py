import tensorflow as tf
from ..l2_normalization import L2Normalization

from ...configs import default
from ...configs.config import CfgNode as CN

from ..partial_transport import MovingAverageStandardization as BatchNormalization

from . import register_pooling

# global pooling
GSoP_cfg = CN()
GSoP_cfg.attention_dim = 128
GSoP_cfg.l2_normalize = False

default.cfg.model.embedding_head.GSoP = GSoP_cfg

@register_pooling(name='GSoP')
@tf.keras.utils.register_keras_serializable()
class GSoP(tf.keras.layers.Layer):
    ''' @InProceedings{Gao_2019_CVPR,
                author = {Zilin, Gao and Jiangtao, Xie and Qilong, Wang and Peihua, Li},
                title = {Global Second-order Pooling Convolutional Networks},
                booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
                year = {2019}
            }'''

    def __init__(self,
                 embedding_size,
                 l2_normalize=True,
                 attention_dim=128,
                 name=None,
                 **kwargs):
        super(GSoP, self).__init__(name=name, **kwargs)

        self._embedding_size = embedding_size
        self._l2_normalize = l2_normalize
        self._attention_dim = attention_dim


    def build(self, input_shape):

        b, h, w, d = input_shape

        d = self._embedding_size

        self._transform = tf.keras.layers.Dense(units=self._embedding_size,
                                                use_bias=False,
                                                kernel_constraint=None,
                                                kernel_initializer='glorot_uniform',
                                                name='feature_transform')

        # channel attention
        self._chatt_head = tf.keras.models.Sequential(
            [
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=self._attention_dim,
                                       kernel_size=1, strides=1, padding='same'),
                BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Reshape(target_shape=(h*w, self._attention_dim)),
                tf.keras.layers.Lambda(lambda x: tf.divide(
                    tf.matmul(x, x, transpose_a=True),
                    tf.cast(tf.square(h * w), tf.float32)
                )
                                       ), # covariance pool for channels
                tf.keras.layers.Reshape(target_shape=(1, self._attention_dim, self._attention_dim)), # shape: (b,1,d,d)
                BatchNormalization(),
                tf.keras.layers.Conv2D(filters=4*self._attention_dim,
                                       kernel_size=(1, self._attention_dim),
                                       strides=1, padding='valid',
                                       groups=self._attention_dim),
                tf.keras.layers.Conv2D(filters=d,
                                       kernel_size=1, strides=1, padding='same',
                                       activation='sigmoid')
            ],
            name='channel_attention'
        )   # output shape: (b, 1, 1, d)

        # spatial attention head
        self._spatt_head = tf.keras.models.Sequential(
            [
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=self._attention_dim,
                                       kernel_size=1, strides=1, padding='same'),
                BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Reshape(target_shape=(h * w, self._attention_dim)),
                tf.keras.layers.Lambda(lambda x: tf.divide(
                    tf.matmul(x, x, transpose_b=True),
                    tf.cast(tf.square(d), tf.float32)
                )
                                       ),  # covariance pool for locations (i.e. shape: (b, wh, wh)
                tf.keras.layers.Reshape(target_shape=(1, w * h, w * h)),  # shape: (b,1,wh,wh)
                BatchNormalization(),
                tf.keras.layers.Conv2D(filters=4*w*h,
                                       kernel_size=(1, w * h),
                                       strides=1, padding='valid',
                                       groups=w*h,
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=w*h,
                                       kernel_size=1, strides=1, padding='same',
                                       activation='sigmoid'),
                tf.keras.layers.Reshape(target_shape=(w, h, 1))
            ],
            name='spatial_attention'
        )  # output shape: (b, w, h, 1)

        self._fuse = tf.keras.layers.Maximum()

        self._gap = tf.keras.layers.GlobalAveragePooling2D()

        self._maybe_normalize = L2Normalization() if self._l2_normalize else \
            tf.keras.layers.Lambda(function=lambda x: x)




    def call(self, inputs, **kwargs):

        features = self._transform(inputs)

        # channel attention

        channel_att = self._chatt_head(features)

        channel_masked_features = tf.multiply(features, channel_att)

        # spatial attention

        spatial_att = self._spatt_head(features)

        location_masked_features = tf.multiply(tf.nn.relu(features), spatial_att)

        weighted_features = self._fuse([channel_masked_features, location_masked_features])

        # no need a residual addition owing to no more blocks to follow
        #enhanced_features = features + weighted_features

        x_pooled = self._gap(weighted_features)

        x_emb = self._maybe_normalize(x_pooled)

        return x_emb


    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], self._embedding_size)

        return output_shape

    def get_config(self):
        config = super(GSoP, self).get_config()
        config.update({'embedding_size': self._embedding_size,
                       'l2_normalize': self._l2_normalize,
                       'attention_dim': self._attention_dim
                       }
                      )
        return config



