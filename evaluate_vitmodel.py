import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from models.cnn_vit_model import PositionalEmbedding,TransformerBlock  # Register custom layer
from utils.data_loader import get_data_generators

# Load model with custom layer
model = tf.keras.models.load_model(
    'saved_models/vit_dataset-1.h5',
    custom_objects={
        'TransformerBlock': TransformerBlock,
        'PositionalEmbedding': PositionalEmbedding  
    }
)

# Load test data
_, _, test_gen = get_data_generators("dataset1")
class_names = list(test_gen.class_indices.keys())

# Predict on test data
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes


# Classification report
print("\n📝 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
