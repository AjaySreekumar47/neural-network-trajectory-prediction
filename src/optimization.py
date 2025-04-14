# First, make sure the Callback base class is defined
class Callback:
    """Base class for callbacks"""
    
    def on_training_begin(self, model):
        """Called at the start of training"""
        pass
    
    def on_training_end(self, history):
        """Called at the end of training"""
        pass
    
    def on_epoch_begin(self, epoch, model):
        """Called at the start of an epoch"""
        pass
    
    def on_epoch_end(self, epoch, logs):
        """Called at the end of an epoch"""
        pass
    
    def on_batch_begin(self, batch, model):
        """Called at the start of a batch"""
        pass
    
    def on_batch_end(self, batch, logs):
        """Called at the end of a batch"""
        pass

# Then define EarlyStopping
class EarlyStopping(Callback):
    """Early stopping callback"""
    
    def __init__(self, monitor='val_loss', patience=10, min_delta=0, verbose=1):
        """Initialize early stopping callback."""
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.wait = 0
        self.best = float('inf') if 'loss' in monitor else -float('inf')
        self.stop_training = False
    
    def on_epoch_end(self, epoch, logs):
        """Check for early stopping condition at the end of epoch."""
        current = logs.get(self.monitor)
        if current is None:
            print(f"Warning: {self.monitor} is not available in logs")
            return
        
        if 'loss' in self.monitor:
            # For loss metrics, improvement is decrease
            if current < self.best - self.min_delta:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stop_training = True
                    if self.verbose:
                        print(f"Early stopping: {self.monitor} did not improve for {self.patience} epochs")
        else:
            # For other metrics like accuracy, improvement is increase
            if current > self.best + self.min_delta:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stop_training = True
                    if self.verbose:
                        print(f"Early stopping: {self.monitor} did not improve for {self.patience} epochs")

# And define ModelCheckpoint 
class ModelCheckpoint(Callback):
    """Model checkpoint callback to save the best model"""
    
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, verbose=1):
        """Initialize model checkpoint callback."""
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = float('inf') if 'loss' in monitor else -float('inf')
        self.model = None
    
    def on_training_begin(self, model):
        """Set the model reference at the beginning of training."""
        self.model = model
    
    def on_epoch_end(self, epoch, logs):
        """Check if model should be saved at the end of epoch."""
        current = logs.get(self.monitor)
        if current is None:
            print(f"Warning: {self.monitor} is not available in logs")
            return
        
        if self.save_best_only:
            if ('loss' in self.monitor and current < self.best) or \
               ('loss' not in self.monitor and current > self.best):
                if self.verbose:
                    print(f"Epoch {epoch+1}: {self.monitor} improved from {self.best:.6f} to {current:.6f}, saving model to {self.filepath}")
                self.best = current
                
                # Save the model
                import pickle
                with open(self.filepath, 'wb') as f:
                    pickle.dump(self.model, f)
            else:
                if self.verbose:
                    print(f"Epoch {epoch+1}: {self.monitor} did not improve from {self.best:.6f}")

filepath = os.path.join(config.get('models_path', './models/'), f"{config['type']}_model.pkl")

class Optimizer:
    """Base class for optimizers"""
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize optimizer with learning rate.
        
        Args:
            learning_rate: Initial learning rate
        """
        self.learning_rate = learning_rate
        self.iterations = 0
    
    def update(self, layer):
        """
        Update the weights of a layer.
        
        Args:
            layer: The layer to update
        """
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum factor (0 = no momentum)
            nesterov: Whether to use Nesterov momentum
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
    
    def update(self, layer):
        """
        Update weights using SGD with optional momentum.
        
        Args:
            layer: Layer to update
        """
        if not hasattr(layer, 'weights') or not hasattr(layer, 'bias'):
            return
        
        # Initialize momentum if not already done
        if not hasattr(layer, 'weight_momentum'):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.bias)
        
        # Calculate gradients (these should be set during backpropagation)
        if not hasattr(layer, 'weight_gradient'):
            layer.weight_gradient = np.zeros_like(layer.weights)
            layer.bias_gradient = np.zeros_like(layer.bias)
        
        # Update with momentum
        layer.weight_momentum = self.momentum * layer.weight_momentum - self.learning_rate * layer.weight_gradient
        layer.bias_momentum = self.momentum * layer.bias_momentum - self.learning_rate * layer.bias_gradient
        
        if self.nesterov:
            # Nesterov update
            layer.weights += self.momentum * layer.weight_momentum - self.learning_rate * layer.weight_gradient
            layer.bias += self.momentum * layer.bias_momentum - self.learning_rate * layer.bias_gradient
        else:
            # Standard momentum update
            layer.weights += layer.weight_momentum
            layer.bias += layer.bias_momentum


class RMSprop(Optimizer):
    """RMSprop optimizer"""
    
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        """
        Initialize RMSprop optimizer.
        
        Args:
            learning_rate: Learning rate
            decay_rate: Decay rate for moving average of squared gradients
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
    
    def update(self, layer):
        """
        Update weights using RMSprop.
        
        Args:
            layer: Layer to update
        """
        if not hasattr(layer, 'weights') or not hasattr(layer, 'bias'):
            return
        
        # Initialize cache if not already done
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)
        
        # Calculate gradients (these should be set during backpropagation)
        if not hasattr(layer, 'weight_gradient'):
            layer.weight_gradient = np.zeros_like(layer.weights)
            layer.bias_gradient = np.zeros_like(layer.bias)
        
        # Update cache
        layer.weight_cache = self.decay_rate * layer.weight_cache + (1 - self.decay_rate) * (layer.weight_gradient ** 2)
        layer.bias_cache = self.decay_rate * layer.bias_cache + (1 - self.decay_rate) * (layer.bias_gradient ** 2)
        
        # Update weights and biases
        layer.weights -= self.learning_rate * layer.weight_gradient / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.bias -= self.learning_rate * layer.bias_gradient / (np.sqrt(layer.bias_cache) + self.epsilon)


class Adam(Optimizer):
    """Adam optimizer"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0
    
    def update(self, layer):
        """
        Update weights using Adam.
        
        Args:
            layer: Layer to update
        """
        if not hasattr(layer, 'weights') or not hasattr(layer, 'bias'):
            return
        
        # Initialize moment estimates if not already done
        if not hasattr(layer, 'weight_m'):
            layer.weight_m = np.zeros_like(layer.weights)  # First moment (momentum)
            layer.bias_m = np.zeros_like(layer.bias)
            layer.weight_v = np.zeros_like(layer.weights)  # Second moment (RMSprop)
            layer.bias_v = np.zeros_like(layer.bias)
        
        # Increment iteration counter
        self.iterations += 1
        
        # Calculate gradients (these should be set during backpropagation)
        if not hasattr(layer, 'weight_gradient'):
            layer.weight_gradient = np.zeros_like(layer.weights)
            layer.bias_gradient = np.zeros_like(layer.bias)
        
        # Update biased first moment estimate
        layer.weight_m = self.beta1 * layer.weight_m + (1 - self.beta1) * layer.weight_gradient
        layer.bias_m = self.beta1 * layer.bias_m + (1 - self.beta1) * layer.bias_gradient
        
        # Update biased second raw moment estimate
        layer.weight_v = self.beta2 * layer.weight_v + (1 - self.beta2) * (layer.weight_gradient ** 2)
        layer.bias_v = self.beta2 * layer.bias_v + (1 - self.beta2) * (layer.bias_gradient ** 2)
        
        # Compute bias-corrected first moment estimate
        weight_m_corrected = layer.weight_m / (1 - self.beta1 ** self.iterations)
        bias_m_corrected = layer.bias_m / (1 - self.beta1 ** self.iterations)
        
        # Compute bias-corrected second raw moment estimate
        weight_v_corrected = layer.weight_v / (1 - self.beta2 ** self.iterations)
        bias_v_corrected = layer.bias_v / (1 - self.beta2 ** self.iterations)
        
        # Update parameters
        layer.weights -= self.learning_rate * weight_m_corrected / (np.sqrt(weight_v_corrected) + self.epsilon)
        layer.bias -= self.learning_rate * bias_m_corrected / (np.sqrt(bias_v_corrected) + self.epsilon)


class AdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay)"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        """
        Initialize AdamW optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay factor
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.iterations = 0
    
    def update(self, layer):
        """
        Update weights using AdamW.
        
        Args:
            layer: Layer to update
        """
        if not hasattr(layer, 'weights') or not hasattr(layer, 'bias'):
            return
        
        # Initialize moment estimates if not already done
        if not hasattr(layer, 'weight_m'):
            layer.weight_m = np.zeros_like(layer.weights)  # First moment
            layer.bias_m = np.zeros_like(layer.bias)
            layer.weight_v = np.zeros_like(layer.weights)  # Second moment
            layer.bias_v = np.zeros_like(layer.bias)
        
        # Increment iteration counter
        self.iterations += 1
        
        # Calculate gradients (these should be set during backpropagation)
        if not hasattr(layer, 'weight_gradient'):
            layer.weight_gradient = np.zeros_like(layer.weights)
            layer.bias_gradient = np.zeros_like(layer.bias)
        
        # Update biased first moment estimate
        layer.weight_m = self.beta1 * layer.weight_m + (1 - self.beta1) * layer.weight_gradient
        layer.bias_m = self.beta1 * layer.bias_m + (1 - self.beta1) * layer.bias_gradient
        
        # Update biased second raw moment estimate
        layer.weight_v = self.beta2 * layer.weight_v + (1 - self.beta2) * (layer.weight_gradient ** 2)
        layer.bias_v = self.beta2 * layer.bias_v + (1 - self.beta2) * (layer.bias_gradient ** 2)
        
        # Compute bias-corrected first moment estimate
        weight_m_corrected = layer.weight_m / (1 - self.beta1 ** self.iterations)
        bias_m_corrected = layer.bias_m / (1 - self.beta1 ** self.iterations)
        
        # Compute bias-corrected second raw moment estimate
        weight_v_corrected = layer.weight_v / (1 - self.beta2 ** self.iterations)
        bias_v_corrected = layer.bias_v / (1 - self.beta2 ** self.iterations)
        
        # Update parameters with decoupled weight decay
        # The key difference from Adam is that weight decay is applied directly to the weights
        # rather than to the gradients
        layer.weights -= self.learning_rate * (
            weight_m_corrected / (np.sqrt(weight_v_corrected) + self.epsilon) + 
            self.weight_decay * layer.weights
        )
        
        layer.bias -= self.learning_rate * bias_m_corrected / (np.sqrt(bias_v_corrected) + self.epsilon)


class LearningRateScheduler:
    """Base class for learning rate schedulers"""
    
    def __init__(self, optimizer):
        """
        Initialize scheduler with an optimizer.
        
        Args:
            optimizer: Optimizer instance
        """
        self.optimizer = optimizer
        self.initial_learning_rate = optimizer.learning_rate
    
    def step(self, epoch=None):
        """
        Update learning rate based on current epoch.
        
        Args:
            epoch: Current epoch number
        """
        raise NotImplementedError


class StepDecay(LearningRateScheduler):
    """Step decay learning rate scheduler"""
    
    def __init__(self, optimizer, drop_rate=0.5, epochs_drop=10):
        """
        Initialize step decay scheduler.
        
        Args:
            optimizer: Optimizer instance
            drop_rate: Factor by which to drop learning rate
            epochs_drop: Number of epochs after which to drop learning rate
        """
        super().__init__(optimizer)
        self.drop_rate = drop_rate
        self.epochs_drop = epochs_drop
    
    def step(self, epoch):
        """
        Update learning rate based on current epoch.
        
        Args:
            epoch: Current epoch number
        """
        exponent = np.floor((1 + epoch) / self.epochs_drop)
        new_lr = self.initial_learning_rate * (self.drop_rate ** exponent)
        self.optimizer.learning_rate = new_lr
        return new_lr


class ExponentialDecay(LearningRateScheduler):
    """Exponential decay learning rate scheduler"""
    
    def __init__(self, optimizer, decay_rate=0.96, decay_steps=100):
        """
        Initialize exponential decay scheduler.
        
        Args:
            optimizer: Optimizer instance
            decay_rate: Decay rate
            decay_steps: Decay steps
        """
        super().__init__(optimizer)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
    
    def step(self, epoch):
        """
        Update learning rate based on current epoch.
        
        Args:
            epoch: Current epoch number
        """
        new_lr = self.initial_learning_rate * (self.decay_rate ** (epoch / self.decay_steps))
        self.optimizer.learning_rate = new_lr
        return new_lr


class CosineAnnealingLR(LearningRateScheduler):
    """Cosine annealing learning rate scheduler"""
    
    def __init__(self, optimizer, T_max, eta_min=0):
        """
        Initialize cosine annealing scheduler.
        
        Args:
            optimizer: Optimizer instance
            T_max: Maximum number of iterations
            eta_min: Minimum learning rate
        """
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
    
    def step(self, epoch):
        """
        Update learning rate based on current epoch.
        
        Args:
            epoch: Current epoch number
        """
        new_lr = self.eta_min + (self.initial_learning_rate - self.eta_min) * (
            1 + np.cos(np.pi * epoch / self.T_max)
        ) / 2
        
        self.optimizer.learning_rate = new_lr
        return new_lr


# Enhanced Neural Network class with optimizer support
class EnhancedNeuralNetwork:
    """Enhanced neural network with optimizer support"""
    
    def __init__(self):
        """Initialize an empty neural network"""
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.lr_scheduler = None
        self.loss_history = []
    
    def add(self, layer):
        """
        Add a layer to the network.
        
        Args:
            layer: Layer instance to add
        """
        self.layers.append(layer)
    
    def set_loss(self, loss):
        """
        Set the loss function.
        
        Args:
            loss: Loss function instance
        """
        self.loss = loss
    
    def set_optimizer(self, optimizer):
        """
        Set the optimizer.
        
        Args:
            optimizer: Optimizer instance
        """
        self.optimizer = optimizer
    
    def set_lr_scheduler(self, scheduler):
        """
        Set the learning rate scheduler.
        
        Args:
            scheduler: Learning rate scheduler instance
        """
        self.lr_scheduler = scheduler
    
    def predict(self, input_data):
        """
        Make predictions with the network.
        
        Args:
            input_data: Input data
            
        Returns:
            Network predictions
        """
        # Set all dropout layers to inference mode
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = False
        
        # Forward pass through all layers
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        
        return output
    
    def _compute_gradients(self, x_batch, y_batch):
        """
        Compute gradients for a single batch.
        
        Args:
            x_batch: Input batch
            y_batch: Target batch
            
        Returns:
            Batch loss
        """
        # Forward pass
        output = x_batch
        for layer in self.layers:
            output = layer.forward(output)
        
        # Compute loss
        batch_loss = self.loss(y_batch, output)
        
        # Backward pass
        gradient = self.loss.backward(y_batch, output)
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, 1.0)  # LR will be applied by optimizer
        
        return batch_loss
    
    def _apply_gradients(self):
        """Apply gradients using optimizer"""
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Use set_optimizer method.")
        
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                self.optimizer.update(layer)
    
    def train(self, x_train, y_train, x_val, y_val, epochs, batch_size, callbacks=None):
        """
        Train the neural network.
        
        Args:
            x_train: Training inputs
            y_train: Training targets
            x_val: Validation inputs
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Size of each training batch
            callbacks: List of callback functions
            
        Returns:
            Loss history dictionary
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Use set_optimizer method.")
        
        # Set all dropout layers to training mode
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = True
        
        # History for tracking metrics
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Initialize callbacks
        callbacks = callbacks or []
        for callback in callbacks:
            callback.on_training_begin(self)
        
        # Number of training samples
        n_samples = len(x_train)
        
        # Training loop
        for epoch in range(epochs):
            # Update learning rate if scheduler is set
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.step(epoch)
                history['learning_rate'].append(lr)
            else:
                history['learning_rate'].append(self.optimizer.learning_rate)
            
            # Callback at epoch start
            for callback in callbacks:
                callback.on_epoch_begin(epoch, self)
            
            # Shuffle indices for this epoch
            indices = np.random.permutation(n_samples)
            
            # Track epoch loss
            epoch_loss = 0
            
            # Process mini-batches
            num_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division
            
            for batch_idx in range(num_batches):
                # Callback at batch start
                for callback in callbacks:
                    callback.on_batch_begin(batch_idx, self)
                
                # Get batch indices
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_x = x_train[batch_indices]
                batch_y = y_train[batch_indices]
                
                # Compute gradients
                batch_loss = self._compute_gradients(batch_x, batch_y)
                
                # Apply gradients
                self._apply_gradients()
                
                # Update epoch loss
                batch_weight = len(batch_indices) / n_samples
                epoch_loss += batch_loss * batch_weight
                
                # Callback at batch end
                for callback in callbacks:
                    callback.on_batch_end(batch_idx, {'loss': batch_loss})
            
            # Evaluate on validation set
            val_predictions = self.predict(x_val)
            val_loss = self.loss(y_val, val_predictions)
            
            # Save history
            history['train_loss'].append(epoch_loss)
            history['val_loss'].append(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - lr: {self.optimizer.learning_rate:.6f}")
            
            # Callback at epoch end
            epoch_logs = {'loss': epoch_loss, 'val_loss': val_loss}
            for callback in callbacks:
                callback.on_epoch_end(epoch, epoch_logs)
            
            # Check for early stopping
            if any(isinstance(callback, EarlyStopping) and callback.stop_training for callback in callbacks):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Callback at training end
        for callback in callbacks:
            callback.on_training_end(history)
        
        return history


# Callback system for the neural network
class Callback:
    """Base class for callbacks"""
    
    def on_training_begin(self, model):
        """Called at the start of training"""
        pass
    
    def on_training_end(self, history):
        """Called at the end of training"""
        pass
    
    def on_epoch_begin(self, epoch, model):
        """Called at the start of an epoch"""
        pass
    
    def on_epoch_end(self, epoch, logs):
        """
        Check if model should be saved at the end of epoch.
        
        Args:
            epoch: Current epoch number
            logs: Logs with metrics
        """
        current = logs.get(self.monitor)
        if current is None:
            print(f"Warning: {self.monitor} is not available in logs")
            return
        
        if ('loss' in self.monitor and current < self.best) or \
           ('loss' not in self.monitor and current > self.best):
            if self.verbose:
                print(f"Epoch {epoch+1}: {self.monitor} improved from {self.best} to {current}, saving model to {self.filepath}")
            self.best = current
            
            # Save the model
            # In a real implementation, we would save the model here
            # For our implementation, we'll pickle the model
            import pickle
            with open(self.filepath, 'wb') as f:
                pickle.dump(self.model, f)
        else:
            if self.verbose:
                print(f"Epoch {epoch+1}: {self.monitor} did not improve from {self.best}")


class TensorBoard(Callback):
    """TensorBoard callback for visualization"""
    
    def __init__(self, log_dir='./logs'):
        """
        Initialize TensorBoard callback.
        
        Args:
            log_dir: Directory to save logs
        """
        super().__init__()
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs):
        """
        Save logs at the end of epoch.
        
        Args:
            epoch: Current epoch number
            logs: Logs with metrics
        """
        # In a real TensorBoard implementation, we would write to TensorBoard here
        # For our simple implementation, we'll just save the logs to a file
        with open(os.path.join(self.log_dir, f'epoch_{epoch}.json'), 'w') as f:
            import json
            json.dump(logs, f)


# Example: Creating a trajectory prediction model with enhanced training
def create_enhanced_trajectory_model(input_shape, output_shape, optimizer_config=None):
    """
    Create a neural network for trajectory prediction with enhanced training.
    
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
    
    # Set loss function
    # For trajectory prediction, we might want to weight later time steps more heavily
    time_weights = np.linspace(0.5, 1.5, output_shape[0])  # Increasing weights
    model.set_loss(TrajectoryMSE(time_weights=time_weights))
    
    # Configure optimizer based on provided configuration
    if optimizer_config is None:
        optimizer_config = {
            'type': 'adam',
            'learning_rate': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'schedule': {
                'type': 'exponential',
                'decay_rate': 0.95,
                'decay_steps': 5
            }
        }
    
    # Create optimizer
    optimizer_type = optimizer_config.get('type', 'adam').lower()
    lr = optimizer_config.get('learning_rate', 0.001)
    
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
    
    model.set_optimizer(optimizer)
    
    # Create learning rate scheduler if specified
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
    
    return model


# Testing the enhanced neural network with optimizers
if __name__ == "__main__":
    # Generate small synthetic dataset for testing
    np.random.seed(42)
    X, y = generate_synthetic_trajectory_data(n_samples=100, n_timesteps=10, n_features=4)
    
    # Process the data
    data_processor = DataProcessor(config)
    X_train, y_train, X_val, y_val, X_test, y_test = data_processor.prepare_data(
        X, y, val_split=0.2, test_split=0.1
    )
    
    # Print shapes
    print("Input shape:", X_train.shape)
    print("Output shape:", y_train.shape)
    
    # Create optimizer configurations for testing
    optimizer_configs = [
        {
            'name': 'SGD with Momentum',
            'type': 'sgd',
            'learning_rate': 0.01,
            'momentum': 0.9,
            'nesterov': True
        },
        {
            'name': 'RMSprop',
            'type': 'rmsprop',
            'learning_rate': 0.001,
            'decay_rate': 0.9
        },
        {
            'name': 'Adam',
            'type': 'adam',
            'learning_rate': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'schedule': {
                'type': 'exponential',
                'decay_rate': 0.95,
                'decay_steps': 5
            }
        },
        {
            'name': 'AdamW',
            'type': 'adamw',
            'learning_rate': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.01,
            'schedule': {
                'type': 'cosine',
                't_max': 10,
                'eta_min': 0.0001
            }
        }
    ]
    
    # Test each optimizer
    results = {}
    
    for config in optimizer_configs:
        print(f"\n=== Testing {config['name']} ===")
        
        # Create model with this optimizer config
        model = create_enhanced_trajectory_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            output_shape=(y_train.shape[1], y_train.shape[2]),
            optimizer_config=config
        )
        
        # Create callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001),
            ModelCheckpoint(filepath=os.path.join(config.get('models_path', './models/'), f"{config['type']}_model.pkl"),
        monitor='val_loss')
        ]
        
        # Train the model
        history = model.train(
            x_train=X_train, 
            y_train=y_train,
            x_val=X_val,
            y_val=y_val,
            epochs=10,  # Small number for testing
            batch_size=16,
            callbacks=callbacks
        )
        
        # Evaluate the model
        val_predictions = model.predict(X_val)
        val_loss = model.loss(y_val, val_predictions)
        
        # Store results
        results[config['name']] = {
            'final_val_loss': val_loss,
            'history': history
        }
        
        print(f"Final validation loss: {val_loss:.6f}")
    
    # Compare results
    print("\n=== Optimizer Comparison ===")
    for name, result in results.items():
        print(f"{name}: Final val_loss = {result['final_val_loss']:.6f}")
    
    # Plot learning curves for all optimizers
    plt.figure(figsize=(12, 8))
    
    for name, result in results.items():
        plt.plot(result['history']['val_loss'], label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Optimizer Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("Optimization module testing complete!")

# Completing the Callback Classes for the Optimization Module

class Callback:
    """Base class for callbacks"""
    
    def on_training_begin(self, model):
        """Called at the start of training"""
        pass
    
    def on_training_end(self, history):
        """Called at the end of training"""
        pass
    
    def on_epoch_begin(self, epoch, model):
        """Called at the start of an epoch"""
        pass
    
    def on_epoch_end(self, epoch, logs):
        """Called at the end of an epoch"""
        pass
    
    def on_batch_begin(self, batch, model):
        """Called at the start of a batch"""
        pass
    
    def on_batch_end(self, batch, logs):
        """Called at the end of a batch"""
        pass


class EarlyStopping(Callback):
    """Early stopping callback"""
    
    def __init__(self, monitor='val_loss', patience=10, min_delta=0, verbose=1):
        """
        Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement after which to stop
            min_delta: Minimum change to qualify as improvement
            verbose: Verbosity mode
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.wait = 0
        self.best = float('inf') if 'loss' in monitor else -float('inf')
        self.stop_training = False
    
    def on_epoch_end(self, epoch, logs):
        """
        Check for early stopping condition at the end of epoch.
        
        Args:
            epoch: Current epoch number
            logs: Logs with metrics
        """
        current = logs.get(self.monitor)
        if current is None:
            print(f"Warning: {self.monitor} is not available in logs")
            return
        
        if 'loss' in self.monitor:
            # For loss metrics, improvement is decrease
            if current < self.best - self.min_delta:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stop_training = True
                    if self.verbose:
                        print(f"Early stopping: {self.monitor} did not improve for {self.patience} epochs")
        else:
            # For other metrics like accuracy, improvement is increase
            if current > self.best + self.min_delta:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stop_training = True
                    if self.verbose:
                        print(f"Early stopping: {self.monitor} did not improve for {self.patience} epochs")


class ModelCheckpoint(Callback):
    """Model checkpoint callback to save the best model"""
    
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, verbose=1):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path to save the model
            monitor: Metric to monitor
            save_best_only: Whether to save only the best model
            verbose: Verbosity mode
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = float('inf') if 'loss' in monitor else -float('inf')
        self.model = None
    
    def on_training_begin(self, model):
        """
        Set the model reference at the beginning of training.
        
        Args:
            model: The neural network model
        """
        self.model = model
    
    def on_epoch_end(self, epoch, logs):
        """
        Check if model should be saved at the end of epoch.
        
        Args:
            epoch: Current epoch number
            logs: Logs with metrics
        """
        current = logs.get(self.monitor)
        if current is None:
            print(f"Warning: {self.monitor} is not available in logs")
            return
        
        if self.save_best_only:
            if ('loss' in self.monitor and current < self.best) or \
               ('loss' not in self.monitor and current > self.best):
                if self.verbose:
                    print(f"Epoch {epoch+1}: {self.monitor} improved from {self.best:.6f} to {current:.6f}, saving model to {self.filepath}")
                self.best = current
                
                # Save the model
                import pickle
                with open(self.filepath, 'wb') as f:
                    pickle.dump(self.model, f)
            else:
                if self.verbose:
                    print(f"Epoch {epoch+1}: {self.monitor} did not improve from {self.best:.6f}")
        else:
            # Save model on every epoch if save_best_only is False
            if self.verbose:
                print(f"Epoch {epoch+1}: saving model to {self.filepath}")
            
            # Add epoch number to filename if saving every epoch
            epoch_filepath = self.filepath.replace('.pkl', f'_epoch_{epoch+1}.pkl')
            import pickle
            with open(epoch_filepath, 'wb') as f:
                pickle.dump(self.model, f)


class LearningRateSchedulerCallback(Callback):
    """Learning rate scheduler callback"""
    
    def __init__(self, scheduler, verbose=0):
        """
        Initialize learning rate scheduler callback.
        
        Args:
            scheduler: Learning rate scheduler instance
            verbose: Verbosity mode
        """
        super().__init__()
        self.scheduler = scheduler
        self.verbose = verbose
    
    def on_epoch_begin(self, epoch, model):
        """
        Update learning rate at the start of epoch.
        
        Args:
            epoch: Current epoch number
            model: The neural network model
        """
        new_lr = self.scheduler.step(epoch)
        if self.verbose:
            print(f"Epoch {epoch+1}: Learning rate set to {new_lr:.6f}")


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving"""
    
    def __init__(self, monitor='val_loss', factor=0.1, patience=10, 
                 min_delta=1e-4, min_lr=0, verbose=1):
        """
        Initialize ReduceLROnPlateau callback.
        
        Args:
            monitor: Quantity to monitor
            factor: Factor by which to reduce learning rate
            patience: Number of epochs with no improvement after which learning rate will be reduced
            min_delta: Threshold for measuring improvement
            min_lr: Lower bound on the learning rate
            verbose: Verbosity mode
        """
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.verbose = verbose
        self.best = float('inf') if 'loss' in monitor else -float('inf')
        self.wait = 0
        self.model = None
    
    def on_training_begin(self, model):
        """
        Set the model reference at the beginning of training.
        
        Args:
            model: The neural network model
        """
        self.model = model
    
    def on_epoch_end(self, epoch, logs):
        """
        Check for learning rate reduction at the end of epoch.
        
        Args:
            epoch: Current epoch number
            logs: Logs with metrics
        """
        current = logs.get(self.monitor)
        if current is None:
            print(f"Warning: {self.monitor} is not available in logs")
            return
        
        if 'loss' in self.monitor:
            # For loss metrics, improvement is decrease
            if current < self.best - self.min_delta:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = self.model.optimizer.learning_rate
                    if old_lr > self.min_lr:
                        new_lr = max(old_lr * self.factor, self.min_lr)
                        self.model.optimizer.learning_rate = new_lr
                        if self.verbose:
                            print(f"Epoch {epoch+1}: ReduceLROnPlateau reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")
                        self.wait = 0
        else:
            # For other metrics like accuracy, improvement is increase
            if current > self.best + self.min_delta:
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = self.model.optimizer.learning_rate
                    if old_lr > self.min_lr:
                        new_lr = max(old_lr * self.factor, self.min_lr)
                        self.model.optimizer.learning_rate = new_lr
                        if self.verbose:
                            print(f"Epoch {epoch+1}: ReduceLROnPlateau reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")
                        self.wait = 0


class History(Callback):
    """Callback that records events into a `History` object"""
    
    def __init__(self):
        """Initialize History callback"""
        super().__init__()
        self.history = {}
        
    def on_training_begin(self, model):
        """Initialize history at the start of training"""
        self.history = {}
        
    def on_epoch_end(self, epoch, logs):
        """
        Record metrics at the end of epoch.
        
        Args:
            epoch: Current epoch number
            logs: Logs with metrics
        """
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)


class ProgressBar(Callback):
    """Callback that displays a progress bar for each epoch"""
    
    def __init__(self):
        """Initialize ProgressBar callback"""
        super().__init__()
        try:
            from tqdm.notebook import tqdm
            self.tqdm = tqdm
        except ImportError:
            from tqdm import tqdm
            self.tqdm = tqdm
        self.progbar = None
        self.num_batches = None
        
    def on_epoch_begin(self, epoch, model):
        """
        Initialize progress bar at the start of epoch.
        
        Args:
            epoch: Current epoch number
            model: The neural network model
        """
        print(f"Epoch {epoch+1}")
        self.progbar = self.tqdm(total=self.num_batches, unit='batch')
        
    def on_batch_end(self, batch, logs):
        """
        Update progress bar at the end of batch.
        
        Args:
            batch: Current batch number
            logs: Logs with metrics
        """
        self.progbar.update(1)
        if logs:
            self.progbar.set_postfix(loss=logs.get('loss', 0))
            
    def on_epoch_end(self, epoch, logs):
        """
        Close progress bar at the end of epoch.
        
        Args:
            epoch: Current epoch number
            logs: Logs with metrics
        """
        self.progbar.close()
        if logs:
            log_str = " - ".join([f"{key}: {value:.4f}" for key, value in logs.items()])
            print(f"Epoch {epoch+1}: {log_str}")
            
    def on_training_begin(self, model):
        """
        Set number of batches at the start of training.
        
        Args:
            model: The neural network model
        """
        # We would need to calculate this based on dataset size and batch size
        # For now, we'll set it to None and handle this during training
        self.num_batches = None


class MetricsLogger(Callback):
    """Callback that logs metrics to file"""
    
    def __init__(self, filepath='./logs/metrics.csv'):
        """
        Initialize MetricsLogger callback.
        
        Args:
            filepath: Path to the log file
        """
        super().__init__()
        self.filepath = filepath
        self.metrics_file = None
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
    def on_training_begin(self, model):
        """
        Open log file at the start of training.
        
        Args:
            model: The neural network model
        """
        self.metrics_file = open(self.filepath, 'w')
        
    def on_epoch_end(self, epoch, logs):
        """
        Log metrics at the end of epoch.
        
        Args:
            epoch: Current epoch number
            logs: Logs with metrics
        """
        if epoch == 0:
            # Write header
            header = 'epoch,' + ','.join(logs.keys()) + '\n'
            self.metrics_file.write(header)
            
        # Write metrics
        line = str(epoch+1) + ',' + ','.join([str(logs[key]) for key in logs.keys()]) + '\n'
        self.metrics_file.write(line)
        self.metrics_file.flush()
        
    def on_training_end(self, history):
        """
        Close log file at the end of training.
        
        Args:
            history: Training history
        """
        if self.metrics_file:
            self.metrics_file.close()


class GradientClipping(Callback):
    """Callback for gradient clipping"""
    
    def __init__(self, clip_value=5.0):
        """
        Initialize gradient clipping callback.
        
        Args:
            clip_value: Maximum allowed value for gradients
        """
        super().__init__()
        self.clip_value = clip_value
        
    def on_batch_begin(self, batch, model):
        """
        Apply gradient clipping before gradient update.
        
        Args:
            batch: Current batch number
            model: The neural network model
        """
        # We would apply clipping during the backward pass
        # This requires modification to the network's backward method
        # This is just a placeholder implementation
        pass
