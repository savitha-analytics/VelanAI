import pickle
import matplotlib.pyplot as plt # type: ignore

# Load training history
with open("saved_models/vit_dataset-1.h5", "rb") as f:
    history = pickle.load(f)

acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(acc) + 1)

# Print epoch-wise data
print("📊 Epoch-wise Training History:")
for i in range(len(history['accuracy'])):
    print(f"Epoch {i+1:02d} — "
          f"Train Acc: {history['accuracy'][i]:.4f}, "
          f"Val Acc: {history['val_accuracy'][i]:.4f}, "
          f"Train Loss: {history['loss'][i]:.4f}, "
          f"Val Loss: {history['val_loss'][i]:.4f}")

plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'go-', label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'go-', label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
