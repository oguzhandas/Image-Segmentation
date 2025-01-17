from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(img_size,train_image_path,train_mask_path,val_image_path,val_mask_path):
    image_data_proc = ImageDataGenerator(rescale=1./255)
    mask_data_proc = ImageDataGenerator(rescale=1./255)
    
    
    
    train_image_data = image_data_proc.flow_from_directory(
        train_image_path,
        target_size=(img_size, img_size),
        batch_size=4,
        class_mode=None,
        seed=42
    )
    
    train_mask_data = mask_data_proc.flow_from_directory(
        train_mask_path,
        target_size=(img_size, img_size),
        batch_size=4,
        class_mode=None,
        color_mode='grayscale',
        seed=42
    )
    
    
    val_image_data = image_data_proc.flow_from_directory(
        val_image_path,
        target_size=(img_size, img_size),
        batch_size=4,
        class_mode=None,
        seed=42
    )
    
    val_mask_data = mask_data_proc.flow_from_directory(
        val_mask_path,
        target_size=(img_size, img_size),
        batch_size=4,
        class_mode=None,
        color_mode='grayscale',
        seed=42
    )
    
    
    def image_mask_gen(img, mask):
        while True:
            img_batch = next(img)
            mask_batch = next(mask)
            yield img_batch, mask_batch
            
    train_data = image_mask_gen(train_image_data, train_mask_data)
    val_data = image_mask_gen(val_image_data, val_mask_data)
    return train_data, val_data, train_image_data, train_mask_data, val_image_data, val_mask_data