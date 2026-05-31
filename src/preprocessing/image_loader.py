import numpy as np
import os
from typing import Tuple, Generator, List
import cv2
from sklearn.model_selection import train_test_split

class PetImageDataLoader:
    """
    Memory-efficient image loading for pet classification dataset.
    Handles batch generation without loading entire dataset upfront.
    Designed for Oxford-IIIT Pet Dataset or similar pet classification tasks.
    """

    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (128, 128),
                 batch_size: int = 32, validation_split: float = 0.2):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.validation_split = validation_split

        # Scan directory structure assuming: data_dir/breed_name/*.jpg
        self.image_paths, self.labels = self._scan_directory()
        self.num_classes = len(set(self.labels))

        # Stratified split to maintain breed distribution across train/val sets
        self.train_paths, self.val_paths, self.train_labels, self.val_labels = \
            train_test_split(self.image_paths, self.labels,
                           test_size=validation_split,
                           stratify=self.labels,
                           random_state=42)

        print(f"Loaded {len(self.image_paths)} images across {self.num_classes} pet breeds")
        print(f"Train: {len(self.train_paths)} | Val: {len(self.val_paths)}")

    def _scan_directory(self) -> Tuple[List[str], List[int]]:
        """
        Walk directory tree and build image path + label mappings.
        Assumes directory structure: data_dir/breed_0/*.jpg, data_dir/breed_1/*.jpg
        """
        paths, labels = [], []
        class_names = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        for class_name in class_names:
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(class_path, img_name))
                    labels.append(self.class_to_idx[class_name])

        return paths, labels

    def load_image(self, path: str, normalize: bool = True) -> np.ndarray:
        """
        Load single image with error handling for corrupted files.
        Converts to RGB if needed (handles grayscale and RGBA).
        """
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        # OpenCV loads as BGR, convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)

        if normalize:
            # Scale to [0, 1] range - critical for gradient stability in training
            img = img.astype(np.float32) / 255.0

        return img

    def batch_generator(self, paths: List[str], labels: List[int],
                       shuffle: bool = True, augment: bool = False) -> Generator:
        """
        Yield batches without loading full dataset into memory.
        Used for training loop to handle datasets larger than available RAM.
        """
        num_samples = len(paths)
        indices = np.arange(num_samples)

        while True:  # Infinite generator for training epochs
            if shuffle:
                np.random.shuffle(indices)

            for start_idx in range(0, num_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]

                # Load batch on-demand
                batch_images = []
                for i in batch_indices:
                    img = self.load_image(paths[i])
                    if augment:
                        img = self._augment_image(img)
                    batch_images.append(img)

                batch_images = np.array(batch_images)
                batch_labels = np.array([labels[i] for i in batch_indices])

                yield batch_images, batch_labels

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """
        Simple data augmentation without external libraries.
        Randomly apply horizontal flip and slight rotation.
        """
        # Horizontal flip (50% chance)
        if np.random.rand() > 0.5:
            img = np.fliplr(img)

        # Random rotation (-15 to +15 degrees)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))

        return img

    def get_train_generator(self, augment: bool = True) -> Generator:
        """Training generator with data augmentation enabled."""
        return self.batch_generator(self.train_paths, self.train_labels,
                                   shuffle=True, augment=augment)

    def get_val_generator(self) -> Generator:
        """Validation generator without augmentation (deterministic evaluation)."""
        return self.batch_generator(self.val_paths, self.val_labels,
                                   shuffle=False, augment=False)

    def load_full_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load entire dataset into memory (use only for small datasets).
        Returns: X_train, y_train, X_val, y_val
        """
        X_train = np.array([self.load_image(p) for p in self.train_paths])
        y_train = np.array(self.train_labels)
        X_val = np.array([self.load_image(p) for p in self.val_paths])
        y_val = np.array(self.val_labels)

        return X_train, y_train, X_val, y_val
