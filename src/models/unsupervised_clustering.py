import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import cv2
import os

class UnsupervisedPetClustering:
    """
    Unsupervised clustering using transfer learning features from VGG16.
    Groups pet images by visual similarity without using breed labels.
    Useful for discovering natural groupings in unlabeled pet datasets.
    """

    def __init__(self, n_clusters: int = 10, use_pca: bool = True, pca_components: int = 128):
        self.n_clusters = n_clusters
        self.use_pca = use_pca
        self.pca_components = pca_components

        # Load VGG16 without top classification layer
        # Using pre-trained ImageNet weights as feature extractor (transfer learning)
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        # Extract features from second-to-last layer for richer representations
        self.feature_extractor = Model(inputs=base_model.input,
                                      outputs=base_model.get_layer('block5_pool').output)

        self.kmeans = None
        self.pca = None

    def extract_features(self, image_paths: list, batch_size: int = 32) -> np.ndarray:
        """
        Extract deep features from images using VGG16 convolutional layers.
        Returns flattened feature vectors from final conv block.
        """
        features = []

        print(f"Extracting features from {len(image_paths)} images...")
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            for path in batch_paths:
                img = cv2.imread(path)
                if img is None:
                    print(f"Warning: Failed to load {path}, skipping...")
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))  # VGG16 expects 224x224 input
                batch_images.append(img)

            if len(batch_images) == 0:
                continue

            batch_images = np.array(batch_images)
            batch_images = preprocess_input(batch_images)  # VGG-specific normalization

            # Extract features (no gradients needed for inference)
            batch_features = self.feature_extractor.predict(batch_images, verbose=0)
            # Flatten spatial dimensions: (batch, 7, 7, 512) -> (batch, 25088)
            batch_features = batch_features.reshape(batch_features.shape[0], -1)
            features.append(batch_features)

            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_images)}/{len(image_paths)} images")

        all_features = np.vstack(features)
        print(f"Feature extraction complete: shape {all_features.shape}")
        return all_features

    def fit(self, features: np.ndarray) -> tuple:
        """
        Fit K-means clustering with optional PCA dimensionality reduction.
        Returns cluster assignments and quality metrics.
        """
        # PCA reduces memory footprint and can improve clustering by removing noise
        if self.use_pca:
            print(f"Applying PCA: {features.shape[1]} -> {self.pca_components} dimensions")
            self.pca = PCA(n_components=self.pca_components, random_state=42)
            features_reduced = self.pca.fit_transform(features)
            explained_var = self.pca.explained_variance_ratio_.sum()
            print(f"PCA explained variance: {explained_var:.3f}")
        else:
            features_reduced = features

        # K-means with multiple random initializations to avoid local minima
        print(f"Fitting K-means with {self.n_clusters} clusters...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            max_iter=300,
            random_state=42,
            verbose=0
        )

        cluster_labels = self.kmeans.fit_predict(features_reduced)

        # Compute clustering quality metrics
        silhouette = silhouette_score(features_reduced, cluster_labels)
        davies_bouldin = davies_bouldin_score(features_reduced, cluster_labels)

        metrics = {
            'silhouette_score': silhouette,  # Higher is better (range: -1 to 1)
            'davies_bouldin_index': davies_bouldin,  # Lower is better
            'inertia': self.kmeans.inertia_  # Within-cluster sum of squares
        }

        # Print cluster distribution
        unique, counts = np.unique(cluster_labels, return_counts=True)
        print("\nCluster distribution:")
        for cluster_id, count in zip(unique, counts):
            percentage = (count / len(cluster_labels)) * 100
            print(f"  Cluster {cluster_id}: {count} images ({percentage:.1f}%)")

        return cluster_labels, metrics

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Assign new samples to nearest cluster."""
        if self.kmeans is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        if self.use_pca:
            features = self.pca.transform(features)
        return self.kmeans.predict(features)

    def elbow_method(self, features: np.ndarray, k_range: range) -> dict:
        """
        Run K-means for different K values to find optimal number of clusters.
        Returns inertia values for elbow plot analysis.
        """
        if self.use_pca:
            features = self.pca.fit_transform(features)

        print(f"Running elbow method for K in range {k_range.start} to {k_range.stop-1}")
        inertias = {}
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42, verbose=0)
            kmeans.fit(features)
            inertias[k] = kmeans.inertia_
            print(f"  K={k}: inertia={kmeans.inertia_:.2f}")

        return inertias

    def get_cluster_samples(self, image_paths: list, cluster_labels: np.ndarray,
                          cluster_id: int, n_samples: int = 5) -> list:
        """
        Get sample image paths from a specific cluster.
        Useful for visualizing what the model grouped together.
        """
        indices = np.where(cluster_labels == cluster_id)[0]
        if len(indices) == 0:
            return []

        # Sample random images from cluster
        sample_indices = np.random.choice(indices, size=min(n_samples, len(indices)),
                                         replace=False)
        return [image_paths[i] for i in sample_indices]
