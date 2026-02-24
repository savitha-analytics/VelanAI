import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore
import cv2

from models.cnn_vit_model import TransformerBlock  # Ensure custom layer is imported

# Load model
model = tf.keras.models.load_model(
    "saved_models/best_model_vit.h5",
    custom_objects={'TransformerBlock': TransformerBlock}
)

# Class labels (must match training)
class_names = ['BlackPoint', 'FusariumFootRot', 'HealthyLeaf', 'LeafBlight', 'WheatBlast']

# 🔧 Preprocess image with normalization and optional enhancement
def preprocess_image(image_path, target_size=(256, 256)):
    # Load image using Keras
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)

    # Normalize and enhance
    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
    img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
    img_array = cv2.resize(img_array, target_size)
    img_array = img_array / 255.0

    return np.expand_dims(img_array, axis=0)

# 🎯 Predict and draw confusion matrix for single image
def predict_and_plot_confusion(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]

    # Simulate 1-image confusion matrix (pred = true)
    cm = confusion_matrix([predicted_index], [predicted_index], labels=range(len(class_names)))

    # Plot and save to static/cm.png
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Single Image)")
    plt.tight_layout()
    plt.savefig("static/cm.png")
    plt.close()

    return predicted_class
