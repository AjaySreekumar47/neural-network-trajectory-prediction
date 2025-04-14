"""Neural Network Implementation for Autonomous Systems.

A custom neural network architecture for trajectory prediction with
optimized gradient descent algorithms and regularization techniques.
"""

__version__ = '0.1.0'

# Import main components to make them available when importing the package
from .data_processing import DataProcessor, generate_synthetic_trajectory_data
from .neural_network import NeuralNetwork, EnhancedNeuralNetwork, Layer, Dense, Recurrent, Activation
from .optimization import Optimizer, SGD, RMSprop, Adam, AdamW
from .regularization import Dropout, BatchNormalization, L1Regularizer, L2Regularizer
from .evaluation import EvaluationMetrics, TrajectoryVisualizer, ModelComparison, BaselineModel

# Convenience function to create a pre-configured model
from .utils import create_trajectory_prediction_model, configure_best_model
