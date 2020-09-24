
from Dropblock import *
from keras.layers import *
from  attention_module import *
def BatchActivate(x):
    x = BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
                           beta_initializer='zero', gamma_initializer='one')(x)
    x = Activation('relu')(x)
    return x


def convolution_block_dropblock(x, filters, size, strides=(1, 1), padding='same', activation=True,keep_prob=0.9,block_size=7):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = DropBlock2D(block_size=block_size,keep_prob=keep_prob)(x)
    if activation:
        x = BatchActivate(x)
    return x

def residual_drop_block(blockInput, num_filters=16, batch_activate=False,keep_prob=0.9,block_size=7):
    x = BatchActivate(blockInput)
    x = convolution_block_dropblock(x, num_filters, (3, 3),keep_prob=keep_prob,block_size=block_size)
    x = convolution_block_dropblock(x, num_filters, (3, 3), activation=False,keep_prob=keep_prob,block_size=block_size)
    if blockInput.get_shape().as_list()[-1] !=  x.get_shape().as_list()[-1]:
        blockInput = Conv2D(num_filters, (1, 1), activation=None, padding="same")(blockInput)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x

def RSAB(input,block_size=7,keep_prob=0.9):
    num_filters= input.get_shape().as_list()[-1]
    f = BatchActivate(input)
    f = convolution_block_dropblock(f, num_filters, (3, 3), keep_prob=keep_prob, block_size=block_size)
    f = convolution_block_dropblock(f, num_filters, (3, 3), activation=False, keep_prob=keep_prob,block_size=block_size)
    x = spatial_attention(f)
    result = add([input,x])
    result = BatchActivate(result)
    return result

