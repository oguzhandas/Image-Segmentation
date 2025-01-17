from tensorflow.keras import layers, Model
import tensorflow as tf
from AGCA import AGCA


def encoder(input_tensor, filters):
    cnv = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(input_tensor)
    cnv = layers.BatchNormalization()(cnv)
    cnv = layers.ReLU()(cnv)
    
    cnv = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(cnv)
    cnv = layers.BatchNormalization()(cnv)
    cnv = layers.ReLU()(cnv)
    
    cnv = AGCA(cnv, reduction=16)
    
    return cnv


def decoder(input_tensor, skip_tensor, filters):
    upsample = layers.UpSampling2D(size=(2, 2))(input_tensor)
    
    upsample = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(upsample)

    if upsample.shape[1] != skip_tensor.shape[1] or upsample.shape[2] != skip_tensor.shape[2]:
        height_diff = skip_tensor.shape[1] - upsample.shape[1]
        width_diff = skip_tensor.shape[2] - upsample.shape[2]
        
        if height_diff > 0 or width_diff > 0:
            skip_tensor = layers.Cropping2D(((0, abs(height_diff)), (0, abs(width_diff))))(skip_tensor)
    
    concat = layers.Concatenate()([upsample, skip_tensor])

    cnv = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(concat)
    cnv = layers.BatchNormalization()(cnv)
    cnv = layers.ReLU()(cnv)
    
    cnv = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(cnv)
    cnv = layers.BatchNormalization()(cnv)
    cnv = layers.ReLU()(cnv)
    
    return cnv


def UNETAGCA(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder Blocks with Pooling
    enc_1 = encoder(inputs, 64)
    pool_1 = layers.MaxPooling2D(pool_size=(2, 2))(enc_1)
    
    enc_2 = encoder(pool_1, 128)
    pool_2 = layers.MaxPooling2D(pool_size=(2, 2))(enc_2)
    
    enc_3 = encoder(pool_2, 256)
    pool_3 = layers.MaxPooling2D(pool_size=(2, 2))(enc_3)
    
    enc_4 = encoder(pool_3, 512)
    pool_4 = layers.MaxPooling2D(pool_size=(2, 2))(enc_4)
    
   
    bneck = encoder(pool_4, 1024)  # Bottleneck Layer
    
    # Decoder Blocks
    dec_4 = decoder(bneck, enc_4, 512)
    dec_3 = decoder(dec_4, enc_3, 256)
    dec_2 = decoder(dec_3, enc_2, 128)
    dec_1 = decoder(dec_2, enc_1, 64)
    
    
    outputs = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(dec_1)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

model = UNETAGCA(input_shape=(128,128,3))
