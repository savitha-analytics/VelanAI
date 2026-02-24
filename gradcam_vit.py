import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import cv2  # type: ignore
from models.cnn_vit_model import TransformerBlock  # Required to load the custom layer

# ✅ Load model with custom object
model = tf.keras.models.load_model(
    'saved_models/vit_dataset-1.h5',
    custom_objects={'TransformerBlock': TransformerBlock}
)

# ✅ Identify the last CNN layer name (check with model.summary() if needed)
last_conv_layer_name = "conv2d_2"  # Output from the third Conv2D layer in your hybrid model

# ✅ CHANGE THIS to a valid test image path
image_path = "D:/Projects/Leaf_Disease/dataset1/Test/HealthyLeaf/rotated_30_316.jpg"  # <-- Replace with an actual image

def preprocess_image(image_path, size=(256, 256)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0, img

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def show_heatmap(heatmap, original_img, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(original_img), 1 - alpha, heatmap_color, alpha, 0)

    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title("Grad-CAM Heatmap")
    plt.show()

if __name__ == "__main__":
    img_array, original_img = preprocess_image(image_path)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    show_heatmap(heatmap, original_img)
