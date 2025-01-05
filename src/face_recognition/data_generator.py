import tensorflow as tf
import numpy as np
from utils.image_processing import ImageProcessing
import os
from tqdm import tqdm
import cv2

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, faces_path, batch_size=32, split='train', validation_split=0.2, augment=True):
        """
        Args:
            faces_path: Path to face images
            batch_size: Size of batches
            split: Either 'train' or 'validation'
            validation_split: Fraction of data to use for validation
            augment: Whether to use data augmentation (only in training)
        """
        self.batch_size = batch_size
        self.faces_path = faces_path
        self.split = split
        self.validation_split = validation_split
        self.augment = augment and split == 'train'
        self.classes = ["dad", "mom", "dexter", "deedee", "unknown"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all file paths and create train/validation split for each class
        self.files_by_class = self._get_files_by_class()
        
        # Split data into train and validation for each class
        self.files = []
        self.labels = []
        
        for class_name in self.classes:
            class_files = self.files_by_class[class_name]
            split_idx = int(len(class_files) * (1 - validation_split))
            
            if split == 'train':
                self.files.extend(class_files[:split_idx])
                self.labels.extend([self.class_to_idx[class_name]] * split_idx)
            else:  # validation
                self.files.extend(class_files[split_idx:])
                self.labels.extend([self.class_to_idx[class_name]] * (len(class_files) - split_idx))
        
        # Shuffle if training
        if split == 'train':
            indices = np.arange(len(self.files))
            np.random.shuffle(indices)
            self.files = [self.files[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        print(f"{split} set size: {len(self.files)} images")
        
        # Calculate augmentation factor for proper batch sizing
        if self.augment:
            sample_augmented = self.augment_data(ImageProcessing.read_image(self.files[0]))
            self.augment_factor = len(sample_augmented)
        else:
            self.augment_factor = 1

    def _get_files_by_class(self):
        files_by_class = {}
        for class_name in self.classes:
            class_path = os.path.join(self.faces_path, class_name)
            files = []
            for file in tqdm(os.listdir(class_path), desc=f"Reading {class_name}"):
                if file.endswith('.jpg'):
                    files.append(os.path.join(class_path, file))
            files_by_class[class_name] = files
        return files_by_class

    def augment_data(self, image):
        """
        Perform data augmentation on a single image
        Returns list of augmented images including the original
        """
        augmented_images = [image]  # Start with original image
        
        # Horizontal flip
        flipped = cv2.flip(image, 1)
        augmented_images.append(flipped)
        
        # Rotation
        # for angle in [-15, 15]:
        #     center = (image.shape[1] // 2, image.shape[0] // 2)
        #     matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        #     rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
        #     augmented_images.append(rotated)
        
        # # Brightness adjustment
        # bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        # dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
        # augmented_images.extend([bright, dark])
        
        # # Slight zoom
        # zoom_factor = 1.1
        # zoomed = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor)
        # h, w = image.shape[:2]
        # h_crop = int((zoomed.shape[0] - h) / 2)
        # w_crop = int((zoomed.shape[1] - w) / 2)
        # zoomed = zoomed[h_crop:h_crop+h, w_crop:w_crop+w]
        # augmented_images.append(zoomed)
        
        return augmented_images

    def __len__(self):
        return int(np.ceil(len(self.files) * self.augment_factor / self.batch_size))

    def __getitem__(self, idx):
        start_idx = (idx * self.batch_size) // self.augment_factor
        end_idx = ((idx + 1) * self.batch_size) // self.augment_factor
        
        batch_files = self.files[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        
        # Load and preprocess images
        batch_images = []
        final_labels = []
        
        for file_path, label in zip(batch_files, batch_labels):
            try:
                image = ImageProcessing.read_image(file_path)
                if self.augment:
                    augmented = self.augment_data(image)
                    batch_images.extend(augmented)
                    final_labels.extend([label] * len(augmented))
                else:
                    batch_images.append(image)
                    final_labels.append(label)
            except Exception as e:
                print(f"Error loading image {file_path}: {str(e)}")
                continue
        
        # Ensure we don't exceed batch size
        batch_images = batch_images[:self.batch_size]
        final_labels = final_labels[:self.batch_size]
        
        # Convert labels to categorical
        categorical_labels = tf.keras.utils.to_categorical(final_labels, num_classes=len(self.classes))
        
        return np.array(batch_images), categorical_labels

    def on_epoch_end(self):
        """Shuffle data after each epoch if in training mode"""
        if self.split == 'train':
            indices = np.arange(len(self.files))
            np.random.shuffle(indices)
            self.files = [self.files[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]


if __name__ == "__main__":
    train_generator = DataGenerator("../data/cropped_train", split='train', augment=True)
    val_generator = DataGenerator("../data/cropped_train", split='validation', augment=False)
    
    print("Train batches:", len(train_generator))
    print("Validation batches:", len(val_generator))
    
    train_images, train_labels = train_generator[0]
    print("Train batch shape:", train_images.shape, train_labels.shape)
    
    val_images, val_labels = val_generator[0]
    print("Validation batch shape:", val_images.shape, val_labels.shape)
    
    # Show some images
    for i in range(10):
        image = train_images[i]
        label = train_labels[i]
        class_name = train_generator.classes[np.argmax(label)]
        print(f"Image {i} is {class_name}")
        cv2.imshow("Image", image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
    print("Done")