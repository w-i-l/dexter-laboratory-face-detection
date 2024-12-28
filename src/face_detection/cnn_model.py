import tensorflow as tf
import numpy as np
from utils.image_processing import ImageProcessing
from face_detection.dataset import DataSet
from matplotlib import pyplot as plt

class CNNModel:
    def __init__(self, input_shape: tuple[int, int, int]):
        self.input_shape = input_shape
        print("input_shape", input_shape)
        self.model = self.__create_model()
        
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['mse']
        )
        self.model.summary()

    def __create_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        print("inputs", inputs)
        
        x = tf.keras.layers.Rescaling(1./255)(inputs)
        
        x = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def preprocess_data(self, data, labels):
        print(len(data), len(data[0]))
        X = np.array(data, dtype=np.float32)
        print(X.shape)
        y = np.array(labels, dtype=np.float32)
        return X, y

    def train(self, test_data, test_labels, train_data, train_labels, epochs: int = 10, batch_size: int = 32):
        print("train_data", len(train_data), len(train_data[0]), len(train_data[0][0]), len(train_data[0][0][0]))
        X_train, y_train = self.preprocess_data(train_data, train_labels)
        X_test, y_test = self.preprocess_data(test_data, test_labels)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=6,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def save_model(self, path: str):
        self.model.save(path)
    
    def load_model(self, path: str):
        self.model = tf.keras.models.load_model(path)

    def predict(self, image: np.ndarray) -> float:
        import cv2
        image = ImageProcessing.resize_image(image)
        print("image shape", image.shape) 
        image = np.expand_dims(image, axis=0)  # Add batch dimension  
        prediction = self.model.predict(image, verbose=0)
        return round(prediction[0][0], 2)

if __name__ == "__main__":
    model = CNNModel((ImageProcessing.IMAGE_WIDTH, ImageProcessing.IMAGE_HEIGHT, 3))

    dataset = DataSet("../data/cropped_train", "../data/extracted_patches")
    data, labels = dataset.read_faces()
    print("data", len(data), len(data[0]), len(data[0][0]), len(data[0][0][0]))
    non_faces_data, non_faces_labels = dataset.read_non_faces(size=len(data))
    data += non_faces_data
    labels += non_faces_labels
    train_data, test_data, train_labels, test_labels = dataset.split_dataset(data, labels)

    try:
        history = model.train(test_data, test_labels, train_data, train_labels, epochs=20, batch_size=128)
    except KeyboardInterrupt:
        print("Training interrupted")
    
    model.save_model("../models/face_detector.h5")

    # Test the model
    model.load_model("../models/face_detector.h5")
    faces = data[:5]
    non_faces = non_faces_data[:5]

    data = faces + non_faces
    labels = [1] * 5 + [0] * 5
    predictions = [model.predict(image) for image in data]
    print(predictions)

    plt.figure(figsize=(10, 10))
    # add padding between the images
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for index, (image, label, prediction) in enumerate(zip(data, labels, predictions)):
        plt.subplot(2, 5, index + 1)
        plt.imshow(image)
        label = "Face" if prediction > 0.5 else "Non-face"
        plt.title(f"Label: {label}\nPrediction:  %.2f\n True label: {label}" % prediction)
        plt.axis('off')

    plt.show()