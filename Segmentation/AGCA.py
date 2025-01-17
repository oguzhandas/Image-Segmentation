from tensorflow.keras import layers

def AGCA(input_tensor, reduction=16):
    channel = input_tensor.shape[-1]
    
    avg_pool = layers.GlobalAveragePooling2D()(input_tensor)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    fc1 = layers.Conv2D(channel // reduction, kernel_size=1, activation='relu')(avg_pool)
    fc2 = layers.Conv2D(channel, kernel_size=1, activation='sigmoid')(fc1)
    
    return layers.multiply([input_tensor, fc2])