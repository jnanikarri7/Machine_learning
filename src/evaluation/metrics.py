import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Evaluation utilities for both supervised and unsupervised pet classification.
    Provides confusion matrices, classification reports, and clustering quality metrics.
    """

    @staticmethod
    def evaluate_supervised(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list = None) -> dict:
        """
        Comprehensive evaluation for supervised classification.
        Returns precision, recall, f1-score per class.
        """
        # Get predicted classes from probabilities if needed
        if len(y_pred.shape) == 2:
            y_pred = np.argmax(y_pred, axis=1)

        # Classification report
        report = classification_report(y_true, y_pred, target_names=class_names,
                                      output_dict=True, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'accuracy': report['accuracy'],
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_f1': report['weighted avg']['f1-score']
        }

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: list = None,
                            save_path: str = None):
        """
        Visualize confusion matrix as heatmap.
        Useful for identifying which breeds are commonly confused.
        """
        plt.figure(figsize=(12, 10))

        if class_names is None:
            class_names = [f"Class {i}" for i in range(cm.shape[0])]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Pet Breed Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_training_history(history, save_path: str = None):
        """
        Plot training and validation accuracy/loss curves.
        Helps identify overfitting or underfitting issues.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy Over Epochs')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss plot
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss Over Epochs')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_top_k_accuracy(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          k_values: list = [1, 3, 5], save_path: str = None):
        """
        Plot top-k accuracy for different k values.
        Useful for understanding model confidence in predictions.
        """
        accuracies = []

        for k in k_values:
            # Get top-k predictions
            top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
            # Check if true label is in top-k
            correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
            accuracy = correct.mean()
            accuracies.append(accuracy)

        plt.figure(figsize=(8, 6))
        plt.bar([f'Top-{k}' for k in k_values], accuracies, color='steelblue')
        plt.ylabel('Accuracy')
        plt.title('Top-K Accuracy Analysis')
        plt.ylim(0, 1.0)

        # Add percentage labels on bars
        for i, (k, acc) in enumerate(zip(k_values, accuracies)):
            plt.text(i, acc + 0.02, f'{acc*100:.1f}%', ha='center', fontweight='bold')

        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Top-k accuracy plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_elbow_curve(inertias: dict, save_path: str = None):
        """
        Plot elbow curve for K-means clustering.
        Helps determine optimal number of clusters.
        """
        k_values = sorted(inertias.keys())
        inertia_values = [inertias[k] for k in k_values]

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, inertia_values, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
        plt.title('Elbow Method for Optimal K')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Elbow curve saved to {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def print_clustering_metrics(metrics: dict):
        """Print clustering quality metrics in readable format."""
        print("\n=== Clustering Quality Metrics ===")
        print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"  (Range: -1 to 1, higher is better)")
        print(f"Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
        print(f"  (Lower is better, indicates better cluster separation)")
        print(f"Inertia: {metrics['inertia']:.2f}")
        print(f"  (Within-cluster sum of squares, lower is better)")
