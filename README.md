# Pet Image Classification - Supervised & Unsupervised Learning

Machine learning project for classifying pet images using both supervised (CNN) and unsupervised (K-means clustering) approaches. Trained on open-source pet datasets (Oxford-IIIT Pet Dataset or similar).

## Project Overview

This project implements two complementary approaches for pet image analysis:

1. **Supervised Learning**: CNN-based classifier that learns to identify specific pet breeds using labeled training data
2. **Unsupervised Learning**: K-means clustering with VGG16 feature extraction to discover natural groupings in unlabeled pet images

## Features

- **Custom CNN Architecture**: 3-layer convolutional neural network optimized for pet breed classification
- **Transfer Learning**: VGG16 pre-trained on ImageNet for feature extraction
- **Data Augmentation**: Rotation, flipping, and brightness adjustments to improve generalization
- **Memory-Efficient Loading**: Batch generators to handle large datasets without loading everything into RAM
- **Comprehensive Evaluation**: Confusion matrices, top-k accuracy, clustering quality metrics
- **Elbow Method**: Automated K-selection for optimal clustering

## Project Structure

```
Machine_learning/
├── data/
│   ├── raw/              # Original pet images (organized by breed)
│   ├── processed/        # Preprocessed images
│   └── models/           # Saved model checkpoints
├── src/
│   ├── preprocessing/
│   │   └── image_loader.py      # Data loading and augmentation
│   ├── models/
│   │   ├── supervised_cnn.py    # CNN classifier
│   │   └── unsupervised_clustering.py  # K-means clustering
│   └── evaluation/
│       └── metrics.py           # Evaluation utilities
├── notebooks/            # Jupyter notebooks for experiments
├── configs/
│   └── training_config.yaml    # Training configuration
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/jnanikarri7/Machine_learning.git
cd Machine_learning

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

Place your pet images in `data/raw/` with the following structure:

```
data/raw/
├── breed_1/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── breed_2/
│   ├── image_001.jpg
│   └── ...
└── breed_n/
    └── ...
```

## Usage

### Supervised Learning (CNN Classification)

```python
from src.preprocessing.image_loader import PetImageDataLoader
from src.models.supervised_cnn import PetClassifierCNN

# Load dataset
loader = PetImageDataLoader('data/raw', img_size=(128, 128), batch_size=32)

# Create model
model = PetClassifierCNN(
    input_shape=(128, 128, 3),
    num_classes=loader.num_classes,
    learning_rate=0.001
)

# Train
train_gen = loader.get_train_generator(augment=True)
val_gen = loader.get_val_generator()

history = model.train(
    train_gen=train_gen,
    val_gen=val_gen,
    steps_per_epoch=len(loader.train_paths) // 32,
    validation_steps=len(loader.val_paths) // 32,
    epochs=50
)

# Predict single image
image = loader.load_image('path/to/test_image.jpg')
class_id, confidence = model.predict_single(image)
print(f"Predicted breed: {loader.idx_to_class[class_id]} (confidence: {confidence:.2f})")
```

### Unsupervised Learning (Clustering)

```python
from src.models.unsupervised_clustering import UnsupervisedPetClustering
from src.evaluation.metrics import ModelEvaluator

# Create clustering model
clusterer = UnsupervisedPetClustering(
    n_clusters=10,
    use_pca=True,
    pca_components=128
)

# Extract features from images
features = clusterer.extract_features(loader.image_paths, batch_size=32)

# Find optimal K using elbow method
inertias = clusterer.elbow_method(features, k_range=range(3, 16))
ModelEvaluator.plot_elbow_curve(inertias, save_path='elbow_curve.png')

# Fit clustering
cluster_labels, metrics = clusterer.fit(features)
ModelEvaluator.print_clustering_metrics(metrics)

# Get sample images from a cluster
samples = clusterer.get_cluster_samples(loader.image_paths, cluster_labels, cluster_id=0)
```

## Model Architecture

### Supervised CNN

- Input: 128x128 RGB images
- 3 Convolutional blocks (32, 64, 128 filters)
- Batch normalization after each conv layer
- MaxPooling and Dropout for regularization
- Dense classification head with softmax activation

**Total Parameters**: ~6.5M trainable parameters

### Unsupervised Clustering

- Feature extraction: VGG16 (pre-trained on ImageNet)
- Dimensionality reduction: PCA (512 → 128 dimensions)
- Clustering: K-means with 10 clusters
- Quality metrics: Silhouette score, Davies-Bouldin index

## Results

Training results and model checkpoints will be saved to:
- `data/models/best_pet_classifier.h5` - Best supervised model
- `logs/` - TensorBoard training logs
- Generated plots (confusion matrices, training curves, elbow curves)

## Configuration

Edit `configs/training_config.yaml` to customize:
- Image size and batch size
- Learning rate and epochs
- Data augmentation parameters
- Number of clusters for unsupervised learning

## Dataset Sources

This project is designed to work with:
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (37 breeds, ~7,400 images)
- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) (120 breeds, ~20,000 images)
- Custom pet image collections

## License

MIT License

## Author

Jnana Karri - Senior Data Engineer specializing in AWS, PySpark, and ML pipelines
