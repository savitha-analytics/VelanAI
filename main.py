import os
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
from utils.data_loader import get_data_generators
#from models.cnn_vit_model_2 import build_cnn_vit_model  # ✅ use updated model
from models.cnn_vit_swin_model import build_cnn_vit_swin_model
#build_cnn_vit_model  build_cnn_vit_swin_model

if __name__ == "__main__":
    # Step 1: Load dataset
    train_gen, val_gen, test_gen = get_data_generators("dataset4")  # change folder as needed

    num_classes = train_gen.num_classes  # ✅ auto detect
    print("Class Indices:", train_gen.class_indices)
    print("Training Samples:", train_gen.samples)
    print("Validation Samples:", val_gen.samples)
    print("Test Samples:", test_gen.samples)

    # Step 2: Build CNN + ViT hybrid model
    model = build_cnn_vit_swin_model(input_shape=(256, 256, 3), num_classes=num_classes)
    model.summary()


    # Step 3: Set up Callbacks (dynamic save path)
    model_name = f"D:\Leaf Disease\wheat_disease_detection\saved_models\vit_dataset-1.h5"
    checkpoint_path = os.path.join("saved_models", model_name)

    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )

    earlystop_cb = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Step 4: Train Model
    history = model.fit(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        callbacks=[checkpoint_cb, earlystop_cb],
        verbose=1
    )

    # Step 5: Training Summary
    print("\n🟩 Training Completed!")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

    # Step 6: Save training history (dynamic file name)
    history_name = f"history_vit_swin{num_classes}_classes.pkl"
    with open(os.path.join("saved_models", history_name), "wb") as f:
        pickle.dump(history.history, f)

    print(f"✅ Training history saved to saved_models/{history_name}")
