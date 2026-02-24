import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, send_from_directory  # type: ignore
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2

from models.cnn_vit_model import TransformerBlock  # Register custom layer

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
model = tf.keras.models.load_model(
    'saved_models/vit_dataset-1.h5',
    custom_objects={'TransformerBlock': TransformerBlock}
)

# Class labels (update if changed)
class_names = ['BlackPoint', 'FusariumFootRot', 'HealthyLeaf', 'LeafBlight', 'WheatBlast']

# Utility: preprocess uploaded image
def preprocess_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array, img

# Utility: plot confusion matrix for a single prediction
def save_confusion_matrix(true_class, pred_class, save_path='static/cm.png'):
    cm = confusion_matrix([true_class], [pred_class], labels=range(len(class_names)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img_array, original_img = preprocess_image(filepath)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    predicted_label = class_names[predicted_class]

    # Extract true class from filename if possible
    true_label = next((i for i, cls in enumerate(class_names) if cls.lower() in filename.lower()), predicted_class)
    save_confusion_matrix(true_label, predicted_class)

    return render_template(
        'result.html',
        image_path=filename,  # just filename now
        predicted_class=predicted_label,
        cm_path='static/cm.png'
    )

if __name__ == '__main__':
    app.run(debug=True)
