import tensorflow as tf
import numpy as np
from utils.image_processing import ImageProcessing
from face_detection.dataset import DataSet
from matplotlib import pyplot as plt
from face_recognition.data_generator import DataGenerator
import os

class CNNModel:
    def __init__(self, input_shape: tuple[int, int, int]):
        self.input_shape = input_shape
        self.classes = ["dad", "mom", "dexter", "deedee", "unknown"]
        print("input_shape", input_shape)
        self.model = self.__create_model()
        
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',  # Changed for multi-class classification
            metrics=['accuracy']
        )
        self.model.summary()

    def __create_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        print("inputs", inputs)
        
        x = tf.keras.layers.Rescaling(1./255)(inputs)
        
        x = tf.keras.layers.Conv2D(32, 7, kernel_initializer='he_normal')(x)
        # x = tf.keras.layers.Conv2D(32, 7, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Conv2D(64, 7, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Conv2D(128, 5, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal')(x)
        x = tf.keras.layers.Activation('relu')(x)
        # x = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal')(x)
        # x = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        # Changed final layer to output 5 classes with softmax
        outputs = tf.keras.layers.Dense(len(self.classes), activation='softmax')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def preprocess_data(self, data, labels):
        print(len(data), len(data[0]))
        X = np.array(data, dtype=np.float32)
        print(X.shape)
        y = tf.keras.utils.to_categorical(labels, num_classes=len(self.classes))
        return X, y

    def train_with_generator(self, train_generator, val_generator, epochs=10):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=25,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.85,
                patience=3,
                min_lr=1e-9
            )
        ]

        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=30
        )
        return history
    
    def save_model(self, path: str):
        self.model.save(path)
    
    def load_model(self, path: str):
        self.model = tf.keras.models.load_model(path)

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """Predicts class and confidence for a single image"""
        image = ImageProcessing.resize_image(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension  
        predictions = self.model.predict(image, verbose=0)[0]
        class_idx = np.argmax(predictions)
        confidence = predictions[class_idx]
        return self.classes[class_idx], float(confidence)
    
    def predict_batch(self, images: np.ndarray) -> list[tuple[str, float]]:
        """
        Predict on a batch of images efficiently
        
        Args:
            images: Batch of images (B, H, W, C) or single image (H, W, C)
            
        Returns:
            list of tuples (predicted_class, confidence)
        """
        # Handle single image case
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
            
        batch_size = images.shape[0]
        
        # Resize all images in batch if needed
        if images.shape[1:3] != (ImageProcessing.IMAGE_WIDTH, ImageProcessing.IMAGE_HEIGHT):
            resized_batch = np.zeros((batch_size, ImageProcessing.IMAGE_WIDTH, 
                                    ImageProcessing.IMAGE_HEIGHT, 3), dtype=np.uint8)
            for i in range(batch_size):
                resized_batch[i] = ImageProcessing.resize_image(images[i])
            images = resized_batch
        
        # Get predictions for entire batch
        predictions = self.model.predict(images, verbose=0)
        
        # Convert predictions to class labels and confidences
        results = []
        for pred in predictions:
            class_idx = np.argmax(pred)
            confidence = pred[class_idx]
            results.append((self.classes[class_idx], float(confidence)))
        
        # Return single result for single image
        if batch_size == 1:
            return results[0]
        
        return results


if __name__ == "__main__":
    tf.keras.backend.clear_session()
    model = CNNModel((ImageProcessing.IMAGE_WIDTH, ImageProcessing.IMAGE_HEIGHT, 3))

    # Create data generators
    train_generator = DataGenerator(
        faces_path="../data/cropped_train",
        batch_size=256,
        split='train',
        validation_split=0.2,
        augment=True
    )
    
    val_generator = DataGenerator(
        faces_path="../data/cropped_train",
        batch_size=256,
        split='validation',
        validation_split=0.2,
        augment=False
    )

    # try 

    # Test the model
    model.load_model("../models/face_recognizer.h5")
    
    # Load some test images
    test_images = []
    test_labels = []
    for class_name in model.classes:
        class_path = f"../data/cropped_train/{class_name}"
        files = os.listdir(class_path)[345:347]  # Get 2 images per class
        for file in files:
            img = ImageProcessing.read_image(os.path.join(class_path, file))
            test_images.append(img)
            test_labels.append(class_name)
    
    predictions = [model.predict(image) for image in test_images]
    
    # Visualization
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    
    n_cols = 5
    n_rows = (len(test_images) + n_cols - 1) // n_cols
    
    for idx, (image, true_label, (pred_class, conf)) in enumerate(zip(test_images, test_labels, predictions)):
        plt.subplot(n_rows, n_cols, idx + 1)
        plt.imshow(image)
        color = 'green' if pred_class == true_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_class}\nConf: {conf:.2f}", color=color)
        plt.axis('off')

    plt.show()