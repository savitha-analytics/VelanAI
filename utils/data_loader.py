from keras.preprocessing.image import ImageDataGenerator # type: ignore
import os

def get_data_generators(data_dir, target_size=(256, 256), batch_size=32):
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'validation')
    test_path = os.path.join(data_dir, 'test')

    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        train_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = datagen.flow_from_directory(
        val_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = datagen.flow_from_directory(
        test_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator
