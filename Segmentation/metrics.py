import tensorflow as tf

## Dice Loss Definition
def dice_loss(y_true, y_pred, smth=1e-6):
  
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intsct = tf.reduce_sum(y_true_f * y_pred_f)
    dnm = tf.reduce_sum(y_true_f + y_pred_f)
    
    d_coeff = (2. * intsct + smth) / (dnm + smth) 
    return 1 - d_coeff




def recall(y_true, y_pred):
    y_pred_thresholded = tf.cast(y_pred > 0.4, tf.float32)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred_thresholded, tf.float32))
    pos_positives = tf.reduce_sum(tf.cast(y_true, tf.float32))
    recall = tp / (pos_positives + tf.keras.backend.epsilon())
    
    return recall

def precision(y_true, y_pred):
    y_pred_t = tf.cast(y_pred > 0.7, tf.float32)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred_t, tf.float32))
    pred_positives = tf.reduce_sum(tf.cast(y_pred_t, tf.float32))
    precision = tp / (pred_positives + tf.keras.backend.epsilon())
    
    return precision


def accuracy(y_true, y_pred):
    y_pred_p = tf.cast(y_pred > 0.7, tf.float32)
    y_pred_n = tf.cast(y_pred <= 0.4, tf.float32)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred_p, tf.float32))
    tn = tf.reduce_sum(tf.cast((1 - y_true) * y_pred_n, tf.float32))
    fn = tf.reduce_sum(tf.cast((1 - y_true) * y_pred_p, tf.float32))
    fn = tf.reduce_sum(tf.cast(y_true * y_pred_n, tf.float32))
    total = tp + tn + fn + fn
    accuracy = (tp + tn) / (total + tf.keras.backend.epsilon())

    return accuracy
