import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models  # type: ignore

class PositionalEmbedding(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PositionalEmbedding, self).__init__()
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, inputs):
        # Inputs shape: (batch, tokens, projection_dim)
        positions = tf.range(start=0, limit=tf.shape(inputs)[-2], delta=1)
        embedded_positions = self.position_embedding(positions)
        return inputs + embedded_positions


# Vision Transformer block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

# Hybrid CNN + ViT model
def build_cnn_vit_model(input_shape=(256, 256, 3), num_classes=3):
    inputs = layers.Input(shape=input_shape)

    # Step 1: CNN Encoder
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Step 2: Flatten to token sequence
    x = layers.Reshape((-1, 128))(x)

    # Step 3: Transformer Blocks (x3 stacked)
    for _ in range(3):
        x = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256)(x)

    # Step 4: Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    # Step 5: Optimizer + label smoothing + cosine LR
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000
    )

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=['accuracy']
    )

    return model
