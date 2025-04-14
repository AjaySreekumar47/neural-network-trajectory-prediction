"""Utility functions for the neural network trajectory prediction project."""

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# Import necessary components from your modules
from .neural_network import EnhancedNeuralNetwork, Dense, Recurrent, Reshape, Dropout, Tanh, ReLU
from .regularization import BatchNormalization, L2Regularizer, RegularizedDense
from .optimization import Adam, CosineAnnealingLR
from .evaluation import TrajectoryMSE


def create_trajectory_prediction_model(input_shape, output_shape, optimizer_config=None):
    """
    Create a neural network for trajectory prediction with standard configuration.
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        output_shape: Shape of output data (timesteps, output_dims)
        optimizer_config: Optimizer configuration dictionary
        
    Returns:
        Configured neural network
    """
    model = EnhancedNeuralNetwork()
    
    # Recurrent layer for sequence processing
    model.add(Recurrent(input_size=input_shape[1], hidden_size=64))
    model.add(Tanh())
    
    # Dense layers for trajectory prediction
    model.add(Reshape(input_shape=(input_shape[0], 64), output_shape=(input_shape[0] * 64,)))
    model.add(Dense(input_size=input_shape[0] * 64, output_size=128))
    model.add(ReLU())
    model.add(Dropout(rate=0.2))
    
    model.add(Dense(input_size=128, output_size=128))
    model.add(ReLU())
    model.add(Dropout(rate=0.2))
    
    model.add(Dense(input_size=128, output_size=output_shape[0] * output_shape[1]))
    model.add(Reshape(input_shape=(output_shape[0] * output_shape[1],), 
                      output_shape=output_shape))
    
    # Set loss function with time weighting
    time_weights = np.linspace(0.5, 1.5, output_shape[0])  # Increasing weights
    model.set_loss(TrajectoryMSE(time_weights=time_weights))
    
    # Configure optimizer
    if optimizer_config:
        # Use the provided optimizer configuration
        configure_optimizer(model, optimizer_config)
    else:
        # Use default Adam optimizer
        optimizer = Adam(learning_rate=0.001)
        model.set_optimizer(optimizer)
    
    return model


def configure_best_model(input_shape, output_shape):
    """
    Configure the best performing model based on experiments.
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        output_shape: Shape of output data (timesteps, output_dims)
    
    Returns:
        Configured neural network model
    """
    model = EnhancedNeuralNetwork()
    
    # Add layers with best configuration
    model.add(Recurrent(input_size=input_shape[1], hidden_size=128))
    model.add(Tanh())
    
    model.add(Reshape(input_shape=(input_shape[0], 128), output_shape=(input_shape[0] * 128,)))
    
    model.add(RegularizedDense(input_size=input_shape[0] * 128, output_size=256, 
                              regularizer=L2Regularizer(l2=0.001)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(rate=0.3))
    
    model.add(RegularizedDense(input_size=256, output_size=128, 
                              regularizer=L2Regularizer(l2=0.001)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(rate=0.2))
    
    model.add(RegularizedDense(input_size=128, output_size=output_shape[0] * output_shape[1], 
                              regularizer=L2Regularizer(l2=0.001)))
    model.add(Reshape(input_shape=(output_shape[0] * output_shape[1],), 
                      output_shape=output_shape))
    
    # Configure loss function
    time_weights = np.linspace(0.5, 1.5, output_shape[0])
    model.set_loss(TrajectoryMSE(time_weights=time_weights))
    
    # Configure Adam optimizer
    optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
    model.set_optimizer(optimizer)
    
    # Add learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)
    model.set_lr_scheduler(scheduler)
    
    return model


def configure_optimizer(model, optimizer_config):
    """
    Configure optimizer based on configuration dictionary.
    
    Args:
        model: Neural network model
        optimizer_config: Optimizer configuration dictionary
    """
    from .optimization import SGD, RMSprop, Adam, AdamW
    from .optimization import StepDecay, ExponentialDecay, CosineAnnealingLR
    
    # Extract optimizer type and learning rate
    optimizer_type = optimizer_config.get('type', 'adam').lower()
    lr = optimizer_config.get('learning_rate', 0.001)
    
    # Create the appropriate optimizer
    if optimizer_type == 'sgd':
        momentum = optimizer_config.get('momentum', 0.0)
        nesterov = optimizer_config.get('nesterov', False)
        optimizer = SGD(learning_rate=lr, momentum=momentum, nesterov=nesterov)
    elif optimizer_type == 'rmsprop':
        decay_rate = optimizer_config.get('decay_rate', 0.9)
        optimizer = RMSprop(learning_rate=lr, decay_rate=decay_rate)
    elif optimizer_type == 'adam':
        beta1 = optimizer_config.get('beta1', 0.9)
        beta2 = optimizer_config.get('beta2', 0.999)
        optimizer = Adam(learning_rate=lr, beta1=beta1, beta2=beta2)
    elif optimizer_type == 'adamw':
        beta1 = optimizer_config.get('beta1', 0.9)
        beta2 = optimizer_config.get('beta2', 0.999)
        weight_decay = optimizer_config.get('weight_decay', 0.01)
        optimizer = AdamW(learning_rate=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # Set the optimizer
    model.set_optimizer(optimizer)
    
    # Configure learning rate scheduler if specified
    if 'schedule' in optimizer_config:
        schedule_config = optimizer_config['schedule']
        schedule_type = schedule_config.get('type', '').lower()
        
        if schedule_type == 'step':
            drop_rate = schedule_config.get('drop_rate', 0.5)
            epochs_drop = schedule_config.get('epochs_drop', 10)
            scheduler = StepDecay(optimizer, drop_rate=drop_rate, epochs_drop=epochs_drop)
        elif schedule_type == 'exponential':
            decay_rate = schedule_config.get('decay_rate', 0.96)
            decay_steps = schedule_config.get('decay_steps', 100)
            scheduler = ExponentialDecay(optimizer, decay_rate=decay_rate, decay_steps=decay_steps)
        elif schedule_type == 'cosine':
            t_max = schedule_config.get('t_max', 50)
            eta_min = schedule_config.get('eta_min', 0)
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        else:
            scheduler = None
        
        if scheduler is not None:
            model.set_lr_scheduler(scheduler)


def save_model(model, filepath='./models/trajectory_nn.pkl'):
    """
    Save model to file.
    
    Args:
        model: Neural network model
        filepath: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath='./models/trajectory_nn.pkl'):
    """
    Load model from file.
    
    Args:
        filepath: Path to load the model from
        
    Returns:
        Loaded neural network model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
        
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def ensure_directories(config=None):
    """
    Ensure all necessary directories exist.
    
    Args:
        config: Configuration dictionary with paths (optional)
    """
    # Default directories if config not provided
    directories = ['./data/', './models/', './results/']
    
    # Use directories from config if provided
    if config:
        if 'data_path' in config:
            directories.append(config['data_path'])
        if 'models_path' in config:
            directories.append(config['models_path'])
        if 'results_path' in config:
            directories.append(config['results_path'])
    
    # Create directories if they don't exist
    for path in directories:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")


def setup_plotting_style():
    """Set up matplotlib for consistent plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
