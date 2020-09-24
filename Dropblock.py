import keras
import keras.backend as K


class DropBlock1D(keras.layers.Layer):
    """See: https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format=None,
                 **kwargs):
        """Initialize the layer.
        :param block_size: Size for each mask block.
        :param keep_prob: Probability of keeping the original feature.
        :param sync_channels: Whether to use the same dropout for all channels.
        :param data_format: 'channels_first' or 'channels_last' (default).
        :param kwargs: Arguments for parent class.
        """
        super(DropBlock1D, self).__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = keras.engine.base_layer.InputSpec(ndim=3)
        self.supports_masking = True

    def get_config(self):
        config = {'block_size': self.block_size,
                  'keep_prob': self.keep_prob,
                  'sync_channels': self.sync_channels,
                  'data_format': self.data_format}
        base_config = super(DropBlock1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_gamma(self, feature_dim):
        """Get the number of activation units to drop"""
        feature_dim = K.cast(feature_dim, K.floatx())
        block_size = K.constant(self.block_size, dtype=K.floatx())
        return ((1.0 - self.keep_prob) / block_size) * (feature_dim / (feature_dim - block_size + 1.0))

    def _compute_valid_seed_region(self, seq_length):
        positions = K.arange(seq_length)
        half_block_size = self.block_size // 2
        valid_seed_region = K.switch(
            K.all(
                K.stack(
                    [
                        positions >= half_block_size,
                        positions < seq_length - half_block_size,
                    ],
                    axis=-1,
                ),
                axis=-1,
            ),
            K.ones((seq_length,)),
            K.zeros((seq_length,)),
        )
        return K.expand_dims(K.expand_dims(valid_seed_region, axis=0), axis=-1)

    def _compute_drop_mask(self, shape):
        seq_length = shape[1]
        mask = K.random_binomial(shape, p=self._get_gamma(seq_length))
        mask *= self._compute_valid_seed_region(seq_length)
        mask = keras.layers.MaxPool1D(
            pool_size=self.block_size,
            padding='same',
            strides=1,
            data_format='channels_last',
        )(mask)
        return 1.0 - mask

    def call(self, inputs, training=None):

        def dropped_inputs():
            outputs = inputs
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 2, 1])
            shape = K.shape(outputs)
            if self.sync_channels:
                mask = self._compute_drop_mask([shape[0], shape[1], 1])
            else:
                mask = self._compute_drop_mask(shape)
            outputs = outputs * mask *\
                (K.cast(K.prod(shape), dtype=K.floatx()) / K.sum(mask))
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 2, 1])
            return outputs

        return K.in_train_phase(dropped_inputs, inputs, training=training)


class DropBlock2D(keras.layers.Layer):
    """See: https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format=None,
                 **kwargs):
        """Initialize the layer.
        :param block_size: Size for each mask block.
        :param keep_prob: Probability of keeping the original feature.
        :param sync_channels: Whether to use the same dropout for all channels.
        :param data_format: 'channels_first' or 'channels_last' (default).
        :param kwargs: Arguments for parent class.
        """
        super(DropBlock2D, self).__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = keras.engine.base_layer.InputSpec(ndim=4)
        self.supports_masking = True

    def get_config(self):
        config = {'block_size': self.block_size,
                  'keep_prob': self.keep_prob,
                  'sync_channels': self.sync_channels,
                  'data_format': self.data_format}
        base_config = super(DropBlock2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_gamma(self, height, width):
        """Get the number of activation units to drop"""
        height, width = K.cast(height, K.floatx()), K.cast(width, K.floatx())
        block_size = K.constant(self.block_size, dtype=K.floatx())
        return ((1.0 - self.keep_prob) / (block_size ** 2)) *\
               (height * width / ((height - block_size + 1.0) * (width - block_size + 1.0)))

    def _compute_valid_seed_region(self, height, width):
        positions = K.concatenate([
            K.expand_dims(K.tile(K.expand_dims(K.arange(height), axis=1), [1, width]), axis=-1),
            K.expand_dims(K.tile(K.expand_dims(K.arange(width), axis=0), [height, 1]), axis=-1),
        ], axis=-1)
        half_block_size = self.block_size // 2
        valid_seed_region = K.switch(
            K.all(
                K.stack(
                    [
                        positions[:, :, 0] >= half_block_size,
                        positions[:, :, 1] >= half_block_size,
                        positions[:, :, 0] < height - half_block_size,
                        positions[:, :, 1] < width - half_block_size,
                    ],
                    axis=-1,
                ),
                axis=-1,
            ),
            K.ones((height, width)),
            K.zeros((height, width)),
        )
        return K.expand_dims(K.expand_dims(valid_seed_region, axis=0), axis=-1)

    def _compute_drop_mask(self, shape):
        height, width = shape[1], shape[2]
        mask = K.random_binomial(shape, p=self._get_gamma(height, width))
        mask *= self._compute_valid_seed_region(height, width)
        mask = keras.layers.MaxPool2D(
            pool_size=(self.block_size, self.block_size),
            padding='same',
            strides=1,
            data_format='channels_last',
        )(mask)
        return 1.0 - mask

    def call(self, inputs, training=None):

        def dropped_inputs():
            outputs = inputs
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])
            shape = K.shape(outputs)
            if self.sync_channels:
                mask = self._compute_drop_mask([shape[0], shape[1], shape[2], 1])
            else:
                mask = self._compute_drop_mask(shape)
            outputs = outputs * mask *\
                (K.cast(K.prod(shape), dtype=K.floatx()) / K.sum(mask))
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])
            return outputs

        return K.in_train_phase(dropped_inputs, inputs, training=training)




from keras import backend as K
#K.clear_session()
from keras.engine.topology import Layer
from scipy.stats import bernoulli
import copy
import numpy as np
#
# class DropBlock2D(Layer):
#     """
#     Regularization Technique for Convolutional Layers.
#     Pseudocode:
#     1: Input:output activations of a layer (A), block_size, γ, mode
#     2: if mode == Inference then
#     3: return A
#     4: end if
#     5: Randomly sample mask M: Mi,j ∼ Bernoulli(γ)
#     6: For each zero position Mi,j , create a spatial square mask with the center being Mi,j , the width,
#         height being block_size and set all the values of M in the square to be zero (see Figure 2).
#     7: Apply the mask: A = A × M
#     8: Normalize the features: A = A × count(M)/count_ones(M)
#     # Arguments
#         block_size: A Python integer. The size of the block to be dropped.
#         gamma: float between 0 and 1. controls how many activation units to drop.
#     # References
#         - [DropBlock: A regularization method for convolutional networks](
#            https://arxiv.org/pdf/1810.12890v1.pdf)
#     """
#
#     def __init__(self, block_size, keep_prob, **kwargs):
#         super(DropBlock2D, self).__init__(**kwargs)
#         self.block_size = block_size
#         self.keep_prob = keep_prob
#
#     def call(self, x, training=None):
#
#         '''
#         MAKE SURE TO UNCOMMENT BELOW FOR ACTUAL USE
#         '''
#         # During inference, we do not Drop Blocks. (Similar to DropOut)
#         #         if training == None:
#         #             return x
#
#         # Calculate Gamma
#         #feat_size = int(x.shape[-1])
#         feat_size_width = int(x.shape[-3])
#         feat_size_height = int(x.shape[-2])
#         #gamma = ((1 - self.keep_prob) / (self.block_size ** 2)) * (
#         #            (feat_size ** 2) / ((feat_size - self.block_size + 1) ** 2))
#         gamma = ((1 - self.keep_prob) / (self.block_size ** 2)) * (
#                         (feat_size_width * feat_size_height) / ((feat_size_width - self.block_size + 1) * (feat_size_height - self.block_size + 1)))
#
#         padding = self.block_size // 2
#
#         # Randomly sample mask
#         sample = bernoulli.rvs(size=(feat_size_width - (padding * 2), feat_size_height - (padding * 2)), p=gamma)
#
#         # The above code creates a matrix of zeros and samples ones from the distribution
#         # We would like to flip all of these values
#         sample = 1 - sample
#
#         # Pad the mask with ones
#         sample = np.pad(sample, pad_width=padding, mode='constant', constant_values=1)
#
#         # For each 0, create spatial square mask of shape (block_size x block_size)
#         mask = copy.copy(sample)
#         for i in range(feat_size_width):
#             for j in range(feat_size_height):
#                 if sample[i, j] == 0:
#                     mask[i - padding: i + padding + 1, j - padding: j + padding + 1] = 0
#
#         mask = mask.reshape((feat_size_width, feat_size_height, 1))
#         #print("mask:", mask.shape)
#         # Apply the mask
#         x = x * np.repeat(mask, x.shape[-1], 2)
#         #print("x:", x.shape)
#         # Normalize the features
#         count = np.prod(mask.shape)
#         count_ones = np.count_nonzero(mask == 1)
#         x = x * count / count_ones
#
#         return x
#
#     def get_config(self):
#         config = {'block_size': self.block_size,
#                   #'gamma': self.gamma,
#                   #'seed': self.seed
#                   }
#         base_config = super(DropBlock2D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#     def compute_output_shape(self, input_shape):
#         return input_shape