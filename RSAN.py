from  layer import *
from keras.layers import *
from keras.optimizers import *
from keras.models import *
from attention_module import *
def RSANet(input_size=(512, 512, 3), start_neurons=16, keep_prob=0.9,block_size=7,lr=1e-3):
    inputs = Input(input_size)
    conv1 = residual_drop_block(inputs, start_neurons * 1, False, block_size=block_size, keep_prob=keep_prob)
    conv1 = RSAB(conv1, keep_prob=keep_prob, block_size=block_size)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = residual_drop_block(pool1, start_neurons * 2, False, block_size=block_size, keep_prob=keep_prob)
    conv2 = RSAB(conv2, keep_prob=keep_prob, block_size=block_size)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = residual_drop_block(pool2, start_neurons * 4, False, block_size=block_size, keep_prob=keep_prob)
    conv3 = RSAB(conv3, keep_prob=keep_prob, block_size=block_size)
    pool3 = MaxPooling2D((2, 2))(conv3)

    convm = residual_drop_block(pool3, start_neurons * 8, False, block_size=block_size, keep_prob=keep_prob)
    convm = RSAB(convm, keep_prob=keep_prob, block_size=block_size)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])

    uconv3 = residual_drop_block(uconv3, start_neurons * 4, False, block_size=block_size, keep_prob=keep_prob)
    uconv3 = RSAB(uconv3, keep_prob=keep_prob, block_size=block_size)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = residual_drop_block(uconv2, start_neurons * 2, False, block_size=block_size, keep_prob=keep_prob)
    uconv2 = RSAB(uconv2, keep_prob=keep_prob, block_size=block_size)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = residual_drop_block(uconv1, start_neurons * 1, False, block_size=block_size, keep_prob=keep_prob)
    uconv1 = RSAB(uconv1, keep_prob=keep_prob, block_size=block_size)

    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    model = Model(input=inputs, output=output_layer)

    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model


