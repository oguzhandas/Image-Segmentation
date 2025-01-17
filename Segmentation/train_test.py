import tensorflow as tf
from unet_model import UNETAGCA
from ImportData import load_data
from metrics import accuracy, precision, recall, dice_loss
## Set Training Parameters
img_size=128
input_shape = (img_size, img_size, 3)
epochs=50 

## Set Paths
train_image_path = 'Your path to training images folder'
train_mask_path = 'Your path to training masks folder'
val_image_path = 'Your path to validation images folder'
val_mask_path = 'Your path to validation masks folder'


train_data, val_data,train_image_data,train_mask_data,val_image_data,val_mask_data=load_data(img_size,
                                                                                             train_image_path,
                                                                                             train_mask_path,
                                                                                            val_image_path,
                                                                                             val_mask_path)
model=UNETAGCA(input_shape)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=dice_loss,
    metrics=[accuracy,precision,recall])
history=model.fit(
    train_data,
    steps_per_epoch=len(train_image_data),
    epochs=epochs,
    validation_data=val_data,
    validation_steps=len(val_image_data)
)



def image_mask_gen(img, mask):
    while True:
        img_batch = next(img)
        mask_batch = next(mask)
        yield img_batch, mask_batch
test_phase = model.evaluate(
    image_mask_gen(val_image_data, val_mask_data),
    steps=len(val_image_data)
)

loss_test = test_phase[0]
accuracy_test = test_phase[1]
precision_test = test_phase[2]
recall_test = test_phase[3]
F1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test + 1e-7)

print(f"Test Loss: {loss_test}")
print(f"Test Accuracy: {accuracy_test}")
print(f"Test Precision: {precision_test}")
print(f"Test Recall: {recall_test}")
print(f"Test F1-Score: {F1_test}")