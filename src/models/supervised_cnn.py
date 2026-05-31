import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

class PetClassifierCNN:
    """
    Convolutional neural network for supervised pet breed classification.
    Architecture optimized for distinguishing between visually similar breeds.
    """

    def __init__(self, input_shape: tuple, num_classes: int, learning_rate: float = 0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        """
        3-layer CNN with batch normalization and dropout.
        Trade-off: Simpler architecture trains faster but may underfit on complex breed distinctions.
        For production, consider ResNet50 or EfficientNet with transfer learning.
        """
        model = keras.Sequential([
            # Conv block 1: Learn low-level features (fur textures, edges)
            layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=self.input_shape),
            layers.BatchNormalization(),  # Stabilize gradients during training
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),  # Prevent overfitting on limited training samples

            # Conv block 2: Learn mid-level features (facial structures, body shapes)
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Conv block 3: Learn high-level features (breed-specific characteristics)
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            # Dense classification head
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),  # Aggressive dropout before final layer
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Adam with default decay schedule - works well for most image tasks
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',  # Expects integer labels
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_acc')]
        )

        return model

    def train(self, train_gen, val_gen, steps_per_epoch: int,
             validation_steps: int, epochs: int = 50, model_save_path: str = None):
        """
        Train with early stopping and model checkpointing.
        Saves best model based on validation loss (less prone to overfitting than accuracy).
        """
        if model_save_path is None:
            model_save_path = 'data/models/best_pet_classifier.h5'

        # Ensure model directory exists
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Return class probabilities for batch of images."""
        return self.model.predict(images)

    def predict_single(self, image: np.ndarray) -> tuple:
        """
        Predict single image and return (class_id, confidence).
        Input should be preprocessed (normalized, resized).
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        predictions = self.model.predict(image, verbose=0)
        class_id = np.argmax(predictions[0])
        confidence = predictions[0][class_id]

        return class_id, confidence

    def evaluate(self, test_gen, steps: int) -> dict:
        """Evaluate model on test set and return metrics."""
        results = self.model.evaluate(test_gen, steps=steps, verbose=1)
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'top_3_accuracy': results[2]
        }
        return metrics

    def save_model(self, path: str):
        """Save model architecture and weights."""
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load pre-trained model."""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")
