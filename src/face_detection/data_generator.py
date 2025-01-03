
import tensorflow as tf
import numpy as np
from utils.image_processing import ImageProcessing
import os
from tqdm import tqdm

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, faces_path, non_faces_path, batch_size=32, split='train', validation_split=0.2):
        """
        Args:
            faces_path: Path to face images
            non_faces_path: Path to non-face images
            batch_size: Size of batches
            split: Either 'train' or 'validation'
            validation_split: Fraction of data to use for validation
        """
        self.batch_size = batch_size
        self.faces_path = faces_path
        self.non_faces_path = non_faces_path
        self.split = split
        self.validation_split = validation_split
        
        # Get all file paths and create train/validation split
        self.face_files = self._get_face_files()
        self.non_face_files = self._get_non_face_files()
        
        # Balance the dataset
        # min_samples = min(len(self.face_files), len(self.non_face_files))
        # self.face_files = self.face_files[:min_samples]
        # self.non_face_files = self.non_face_files[:min_samples]
        
        # Split data into train and validation
        split_idx_faces = int(len(self.face_files) * (1 - validation_split))
        split_idx_nonfaces = int(len(self.non_face_files) * (1 - validation_split))
        
        if split == 'train':
            self.face_files = self.face_files[:split_idx_faces]
            self.non_face_files = self.non_face_files[:split_idx_nonfaces]
        else:  # validation
            self.face_files = self.face_files[split_idx_faces:]
            self.non_face_files = self.non_face_files[split_idx_nonfaces:]
        
        # Combine files and create labels
        self.files = self.face_files + self.non_face_files
        self.labels = [1] * len(self.face_files) + [0] * len(self.non_face_files)
        
        # Shuffle if training
        if split == 'train':
            indices = np.arange(len(self.files))
            np.random.shuffle(indices)
            self.files = [self.files[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        print(f"{split} set size: {len(self.files)} images")

    def _get_face_files(self):
        face_files = []
        classes = ["dad", "mom", "dexter", "deedee", "unknown"]
        for class_name in classes:
            class_path = os.path.join(self.faces_path, class_name)
            for file in tqdm(os.listdir(class_path), desc=f"Reading {class_name}"):
                if file.endswith('.jpg'):
                    face_files.append(os.path.join(class_path, file))
        return face_files

    def _get_non_face_files(self):
        non_face_files = []
        for file in tqdm(os.listdir(self.non_faces_path), desc="Reading non-faces"):
            if file.endswith('.jpg'):
                non_face_files.append(os.path.join(self.non_faces_path, file))
        return non_face_files

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Load and preprocess images
        batch_images = []
        for file_path in batch_files:
            try:
                image = ImageProcessing.read_image(file_path)
                batch_images.append(image)
            except Exception as e:
                print(f"Error loading image {file_path}: {str(e)}")
                continue
            
        return np.array(batch_images), np.array(batch_labels)

    def on_epoch_end(self):
        """Shuffle data after each epoch if in training mode"""
        if self.split == 'train':
            indices = np.arange(len(self.files))
            np.random.shuffle(indices)
            self.files = [self.files[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
