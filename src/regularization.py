class Regularizer:
    """Base class for regularizers"""
    
    def __call__(self, weights):
        """
        Calculate regularization term.
        
        Args:
            weights: Weight matrix
            
        Returns:
            Regularization term
        """
        return self.calculate_regularization(weights)
    
    def calculate_regularization(self, weights):
        """
        Calculate regularization term.
        
        Args:
            weights: Weight matrix
            
        Returns:
            Regularization term
        """
        raise NotImplementedError
    
    def calculate_gradient(self, weights):
        """
        Calculate gradient of regularization term.
        
        Args:
            weights: Weight matrix
            
        Returns:
            Gradient of regularization term
        """
        raise NotImplementedError


class L1Regularizer(Regularizer):
    """L1 regularization (Lasso)"""
    
    def __init__(self, l1=0.01):
        """
        Initialize L1 regularizer.
        
        Args:
            l1: L1 regularization factor
        """
        self.l1 = l1
    
    def calculate_regularization(self, weights):
        """
        Calculate L1 regularization term.
        
        Args:
            weights: Weight matrix
            
        Returns:
            L1 regularization term
        """
        return self.l1 * np.sum(np.abs(weights))
    
    def calculate_gradient(self, weights):
        """
        Calculate gradient of L1 regularization term.
        
        Args:
            weights: Weight matrix
            
        Returns:
            Gradient of L1 regularization term
        """
        return self.l1 * np.sign(weights)


class L2Regularizer(Regularizer):
    """L2 regularization (Ridge)"""
    
    def __init__(self, l2=0.01):
        """
        Initialize L2 regularizer.
        
        Args:
            l2: L2 regularization factor
        """
        self.l2 = l2
    
    def calculate_regularization(self, weights):
        """
        Calculate L2 regularization term.
        
        Args:
            weights: Weight matrix
            
        Returns:
            L2 regularization term
        """
        return 0.5 * self.l2 * np.sum(weights ** 2)
    
    def calculate_gradient(self, weights):
        """
        Calculate gradient of L2 regularization term.
        
        Args:
            weights: Weight matrix
            
        Returns:
            Gradient of L2 regularization term
        """
        return self.l2 * weights


class ElasticNetRegularizer(Regularizer):
    """Elastic Net regularization (combination of L1 and L2)"""
    
    def __init__(self, l1=0.01, l2=0.01):
        """
        Initialize Elastic Net regularizer.
        
        Args:
            l1: L1 regularization factor
            l2: L2 regularization factor
        """
        self.l1 = l1
        self.l2 = l2
        self.l1_regularizer = L1Regularizer(l1=l1)
        self.l2_regularizer = L2Regularizer(l2=l2)
    
    def calculate_regularization(self, weights):
        """
        Calculate Elastic Net regularization term.
        
        Args:
            weights: Weight matrix
            
        Returns:
            Elastic Net regularization term
        """
        return self.l1_regularizer.calculate_regularization(weights) + \
               self.l2_regularizer.calculate_regularization(weights)
    
    def calculate_gradient(self, weights):
        """
        Calculate gradient of Elastic Net regularization term.
        
        Args:
            weights: Weight matrix
            
        Returns:
            Gradient of Elastic Net regularization term
        """
        return self.l1_regularizer.calculate_gradient(weights) + \
               self.l2_regularizer.calculate_gradient(weights)


# Extended Dense layer with regularization
class RegularizedDense(Dense):
    """Dense layer with explicit regularization"""
    
    def __init__(self, input_size, output_size, regularizer=None):
        """
        Initialize a regularized dense layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
            regularizer: Regularizer instance
        """
        super().__init__(input_size, output_size)
        self.regularizer = regularizer
    
    def forward(self, input_data):
        """
        Forward pass through the dense layer.
        
        Args:
            input_data: Input data of shape (batch_size, input_size)
            
        Returns:
            Output of shape (batch_size, output_size)
        """
        return super().forward(input_data)
    
    def backward(self, output_gradient, learning_rate):
        """
        Backward pass through the dense layer with regularization.
        
        Args:
            output_gradient: Gradient of the loss with respect to the output
            learning_rate: Learning rate for weight updates
            
        Returns:
            Gradient of the loss with respect to the input
        """
        # Compute gradients from output gradient
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        # Add regularization gradient if regularizer is set
        if self.regularizer is not None:
            reg_gradient = self.regularizer.calculate_gradient(self.weights)
            weights_gradient += reg_gradient
        
        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        
        # Compute gradient with respect to input for backpropagation
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        return input_gradient


# Improved Dropout layer
class ImprovedDropout(Dropout):
    """Improved Dropout layer with configurable noise shape"""
    
    def __init__(self, rate, noise_shape=None):
        """
        Initialize an improved dropout layer.
        
        Args:
            rate: Dropout rate (probability of dropping a unit)
            noise_shape: Shape of the dropout mask (None means same as input shape)
        """
        super().__init__(rate)
        self.noise_shape = noise_shape
    
    def forward(self, input_data):
        """
        Forward pass through the dropout layer.
        
        Args:
            input_data: Input data
            
        Returns:
            Output with dropout applied during training, scaled input during inference
        """
        self.input = input_data
        
        if self.training:
            # Create dropout mask with specified noise shape
            if self.noise_shape is None:
                mask_shape = input_data.shape
            else:
                # Broadcast to input shape
                mask_shape = list(input_data.shape)
                for i, dim in enumerate(self.noise_shape):
                    if dim is not None:
                        mask_shape[i] = dim
            
            # Create and scale the mask
            self.mask = np.random.binomial(1, 1 - self.rate, size=mask_shape) / (1 - self.rate)
            self.output = input_data * self.mask
        else:
            # During inference, we don't drop any units
            self.output = input_data
            
        return self.output


# Spatial Dropout for sequences or images (drops entire feature maps)
class SpatialDropout(Dropout):
    """Spatial Dropout layer that drops entire feature maps"""
    
    def __init__(self, rate, spatial_dims=1):
        """
        Initialize a spatial dropout layer.
        
        Args:
            rate: Dropout rate (probability of dropping a feature map)
            spatial_dims: Number of spatial dimensions (1 for sequences, 2 for images)
        """
        super().__init__(rate)
        self.spatial_dims = spatial_dims
    
    def forward(self, input_data):
        """
        Forward pass through the spatial dropout layer.
        
        Args:
            input_data: Input data of shape (batch_size, ..., features)
            
        Returns:
            Output with dropout applied during training, scaled input during inference
        """
        self.input = input_data
        
        if self.training:
            # Get input shape
            input_shape = input_data.shape
            
            # Create dropout mask
            if self.spatial_dims == 1:
                # For sequence data (batch_size, sequence_length, features)
                # Drop entire feature channels across all time steps
                mask_shape = (input_shape[0], 1, input_shape[2])
                mask = np.random.binomial(1, 1 - self.rate, size=mask_shape) / (1 - self.rate)
                # Broadcast mask to all time steps
                self.mask = np.broadcast_to(mask, input_shape)
            
            elif self.spatial_dims == 2:
                # For image data (batch_size, height, width, channels)
                # Drop entire feature maps
                mask_shape = (input_shape[0], 1, 1, input_shape[3])
                mask = np.random.binomial(1, 1 - self.rate, size=mask_shape) / (1 - self.rate)
                # Broadcast mask to all spatial locations
                self.mask = np.broadcast_to(mask, input_shape)
            
            else:
                raise ValueError(f"Unsupported spatial dimensions: {self.spatial_dims}")
            
            self.output = input_data * self.mask
        else:
            # During inference, we don't drop any units
            self.output = input_data
            
        return self.output


# Gaussian Noise layer for regularization
class GaussianNoise(Layer):
    """Gaussian Noise layer for regularization"""
    
    def __init__(self, stddev=0.1):
        """
        Initialize a Gaussian noise layer.
        
        Args:
            stddev: Standard deviation of the noise
        """
        super().__init__()
        self.stddev = stddev
        self.training = True
        self.noise = None
    
    def forward(self, input_data):
        """
        Forward pass through the Gaussian noise layer.
        
        Args:
            input_data: Input data
            
        Returns:
            Output with noise added during training, unchanged input during inference
        """
        self.input = input_data
        
        if self.training:
            # Add Gaussian noise
            self.noise = np.random.normal(0, self.stddev, size=input_data.shape)
            self.output = input_data + self.noise
        else:
            # During inference, we don't add noise
            self.output = input_data
            
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """
        Backward pass through the Gaussian noise layer.
        
        Args:
            output_gradient: Gradient of the loss with respect to the output
            learning_rate: Learning rate (not used in noise layer)
            
        Returns:
            Gradient of the loss with respect to the input
        """
        # Noise layer doesn't change gradients during backpropagation
        return output_gradient


# BatchNormalization layer
class BatchNormalization(Layer):
    """Batch Normalization layer"""
    
    def __init__(self, momentum=0.99, epsilon=1e-8):
        """
        Initialize a batch normalization layer.
        
        Args:
            momentum: Momentum for moving average
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Parameters to be learned
        self.gamma = None  # Scale parameter
        self.beta = None   # Shift parameter
        
        # Moving statistics for inference
        self.running_mean = None
        self.running_var = None
        
        # For backpropagation
        self.input_centered = None
        self.std = None
        self.normalized = None
        
        # Mode
        self.training = True
    
    def initialize(self, input_shape):
        """
        Initialize parameters based on input shape.
        
        Args:
            input_shape: Shape of input
        """
        feature_dim = input_shape[-1]
        
        if self.gamma is None:
            self.gamma = np.ones(feature_dim)
        
        if self.beta is None:
            self.beta = np.zeros(feature_dim)
        
        if self.running_mean is None:
            self.running_mean = np.zeros(feature_dim)
        
        if self.running_var is None:
            self.running_var = np.ones(feature_dim)
    
    def forward(self, input_data):
        """
        Forward pass through the batch normalization layer.
        
        Args:
            input_data: Input data
            
        Returns:
            Normalized and scaled output
        """
        self.input = input_data
        
        # Initialize parameters if needed
        if self.gamma is None:
            self.initialize(input_data.shape)
        
        # Get input shape
        batch_size = input_data.shape[0]
        
        if self.training:
            # Calculate batch mean and variance
            batch_mean = np.mean(input_data, axis=0)
            batch_var = np.var(input_data, axis=0)
            
            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Normalize input
            self.input_centered = input_data - batch_mean
            self.std = np.sqrt(batch_var + self.epsilon)
            self.normalized = self.input_centered / self.std
            
            # Scale and shift
            self.output = self.gamma * self.normalized + self.beta
        else:
            # Use running statistics for inference
            normalized = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            self.output = self.gamma * normalized + self.beta
        
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """
        Backward pass through the batch normalization layer.
        
        Args:
            output_gradient: Gradient of the loss with respect to the output
            learning_rate: Learning rate for parameter updates
            
        Returns:
            Gradient of the loss with respect to the input
        """
        batch_size = output_gradient.shape[0]
        
        # Gradient with respect to gamma and beta
        gamma_gradient = np.sum(output_gradient * self.normalized, axis=0)
        beta_gradient = np.sum(output_gradient, axis=0)
        
        # Update gamma and beta
        self.gamma -= learning_rate * gamma_gradient
        self.beta -= learning_rate * beta_gradient
        
        # Gradient with respect to normalized input
        normalized_gradient = output_gradient * self.gamma
        
        # Gradient with respect to variance
        var_gradient = np.sum(normalized_gradient * self.input_centered, axis=0) * -0.5 * (self.std ** -3)
        
        # Gradient with respect to mean
        mean_gradient = np.sum(normalized_gradient, axis=0) * (-1.0 / self.std) + \
                        var_gradient * np.sum(-2.0 * self.input_centered, axis=0) / batch_size
        
        # Gradient with respect to input
        input_gradient = normalized_gradient / self.std + \
                        var_gradient * 2.0 * self.input_centered / batch_size + \
                        mean_gradient / batch_size
        
        return input_gradient


# WeightDecay regularization as a callback
class WeightDecay(Callback):
    """Weight decay callback for non-adaptive optimizers like SGD"""
    
    def __init__(self, decay_rate=0.001):
        """
        Initialize weight decay callback.
        
        Args:
            decay_rate: Weight decay rate
        """
        super().__init__()
        self.decay_rate = decay_rate
    
    def on_batch_end(self, batch, logs):
        """
        Apply weight decay after each batch.
        
        Args:
            batch: Batch number
            logs: Logs with metrics
        """
        # Apply weight decay to all dense layers
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                layer.weights *= (1 - self.decay_rate)


# Example: Creating a regularized model for trajectory prediction
def create_regularized_trajectory_model(input_shape, output_shape, regularization_config=None):
    """
    Create a neural network for trajectory prediction with regularization.
    
    Args:
        input_shape: Shape of input data (timesteps, features)
        output_shape: Shape of output data (timesteps, output_dims)
        regularization_config: Regularization configuration dictionary
        
    Returns:
        Configured neural network with regularization
    """
    model = EnhancedNeuralNetwork()
    
    # Use default configuration if none provided
    if regularization_config is None:
        regularization_config = {
            'l1': 0.0,
            'l2': 0.001,
            'dropout_rate': 0.2,
            'recurrent_dropout_rate': 0.1,
            'gaussian_noise': 0.1,
            'batch_norm': True
        }
    
    # Create regularizer for dense layers
    l1 = regularization_config.get('l1', 0.0)
    l2 = regularization_config.get('l2', 0.001)
    
    if l1 > 0 and l2 > 0:
        regularizer = ElasticNetRegularizer(l1=l1, l2=l2)
    elif l1 > 0:
        regularizer = L1Regularizer(l1=l1)
    elif l2 > 0:
        regularizer = L2Regularizer(l2=l2)
    else:
        regularizer = None
    
    # Get dropout rates
    dropout_rate = regularization_config.get('dropout_rate', 0.2)
    recurrent_dropout_rate = regularization_config.get('recurrent_dropout_rate', 0.1)
    
    # Get other regularization parameters
    use_batch_norm = regularization_config.get('batch_norm', True)
    gaussian_noise_stddev = regularization_config.get('gaussian_noise', 0.0)
    
    # Build the network with regularization
    
    # Optional: Add Gaussian noise to input
    if gaussian_noise_stddev > 0:
        model.add(GaussianNoise(stddev=gaussian_noise_stddev))
    
    # Recurrent layer with dropout for recurrent connections
    model.add(Recurrent(input_size=input_shape[1], hidden_size=64))
    
    # Optional: Add spatial dropout after recurrent layer
    if recurrent_dropout_rate > 0:
        model.add(SpatialDropout(rate=recurrent_dropout_rate, spatial_dims=1))
    
    model.add(Tanh())
    
    # Reshape for dense layers
    model.add(Reshape(input_shape=(input_shape[0], 64), output_shape=(input_shape[0] * 64,)))
    
    # First dense layer with regularization
    model.add(RegularizedDense(input_size=input_shape[0] * 64, output_size=128, regularizer=regularizer))
    
    # Optional: Add batch normalization
    if use_batch_norm:
        model.add(BatchNormalization())
        
    model.add(ReLU())
    
    # Add dropout
    if dropout_rate > 0:
        model.add(Dropout(rate=dropout_rate))
    
    # Second dense layer with regularization
    model.add(RegularizedDense(input_size=128, output_size=128, regularizer=regularizer))
    
    # Optional: Add batch normalization
    if use_batch_norm:
        model.add(BatchNormalization())
        
    model.add(ReLU())
    
    # Add dropout
    if dropout_rate > 0:
        model.add(Dropout(rate=dropout_rate))
    
    # Output layer
    model.add(RegularizedDense(input_size=128, output_size=output_shape[0] * output_shape[1], 
                              regularizer=regularizer))
    model.add(Reshape(input_shape=(output_shape[0] * output_shape[1],), 
                      output_shape=output_shape))
    
    # Set loss function with time weighting
    time_weights = np.linspace(0.5, 1.5, output_shape[0])  # Increasing weights
    model.set_loss(TrajectoryMSE(time_weights=time_weights))
    
    return model


# Testing the regularization techniques
if __name__ == "__main__":
    # Generate small synthetic dataset for testing
    np.random.seed(42)
    X, y = generate_synthetic_trajectory_data(n_samples=200, n_timesteps=10, n_features=4)
    
    # Process the data
    data_processor = DataProcessor(config)
    X_train, y_train, X_val, y_val, X_test, y_test = data_processor.prepare_data(
        X, y, val_split=0.2, test_split=0.1
    )
    
    # Print shapes
    print("Input shape:", X_train.shape)
    print("Output shape:", y_train.shape)
    
    # Create different regularization configurations for testing
    regularization_configs = [
        {
            'name': 'No Regularization',
            'l1': 0.0,
            'l2': 0.0,
            'dropout_rate': 0.0,
            'recurrent_dropout_rate': 0.0,
            'gaussian_noise': 0.0,
            'batch_norm': False
        },
        {
            'name': 'L2 Regularization',
            'l1': 0.0,
            'l2': 0.001,
            'dropout_rate': 0.0,
            'recurrent_dropout_rate': 0.0,
            'gaussian_noise': 0.0,
            'batch_norm': False
        },
        {
            'name': 'Dropout Only',
            'l1': 0.0,
            'l2': 0.0,
            'dropout_rate': 0.3,
            'recurrent_dropout_rate': 0.1,
            'gaussian_noise': 0.0,
            'batch_norm': False
        },
        {
            'name': 'Batch Normalization Only',
            'l1': 0.0,
            'l2': 0.0,
            'dropout_rate': 0.0,
            'recurrent_dropout_rate': 0.0,
            'gaussian_noise': 0.0,
            'batch_norm': True
        },
        {
            'name': 'Combined Regularization',
            'l1': 0.0001,
            'l2': 0.001,
            'dropout_rate': 0.2,
            'recurrent_dropout_rate': 0.1,
            'gaussian_noise': 0.05,
            'batch_norm': True
        }
    ]
    
    # Create optimizer configuration
    optimizer_config = {
        'type': 'adam',
        'learning_rate': 0.001,
        'beta1': 0.9,
        'beta2': 0.999
    }
    
    # Test each regularization strategy
    results = {}
    
    for config in regularization_configs:
        print(f"\n=== Testing {config['name']} ===")
        
        # Create model with this regularization config
        model = create_regularized_trajectory_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            output_shape=(y_train.shape[1], y_train.shape[2]),
            regularization_config=config
        )
        
        # Set optimizer
        if optimizer_config['type'] == 'adam':
            optimizer = Adam(
                learning_rate=optimizer_config['learning_rate'],
                beta1=optimizer_config['beta1'],
                beta2=optimizer_config['beta2']
            )
        else:
            optimizer = SGD(learning_rate=optimizer_config['learning_rate'])
            
        model.set_optimizer(optimizer)
        
        # Create callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)
        ]
        
        # Train the model
        history = model.train(
            x_train=X_train, 
            y_train=y_train,
            x_val=X_val,
            y_val=y_val,
            epochs=30,  # More epochs to see regularization effects
            batch_size=16,
            callbacks=callbacks
        )
        
        # Evaluate the model
        train_predictions = model.predict(X_train)
        train_loss = model.loss(y_train, train_predictions)
        
        val_predictions = model.predict(X_val)
        val_loss = model.loss(y_val, val_predictions)
        
        # Calculate overfitting metric
        overfitting_ratio = val_loss / train_loss
        
        # Store results
        results[config['name']] = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'overfitting_ratio': overfitting_ratio,
            'history': history
        }
        
        print(f"Train loss: {train_loss:.6f}")
        print(f"Validation loss: {val_loss:.6f}")
        print(f"Overfitting ratio (val/train): {overfitting_ratio:.4f}")
    
    # Compare regularization strategies
    print("\n=== Regularization Comparison ===")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Train loss: {result['train_loss']:.6f}")
        print(f"  Val loss: {result['val_loss']:.6f}")
        print(f"  Overfitting ratio: {result['overfitting_ratio']:.4f}")
    
    # Find baseline for comparison
    baseline_result = results['No Regularization']
    baseline_overfitting = baseline_result['overfitting_ratio']
    
    # Calculate improvement percentages
    print("\n=== Overfitting Reduction ===")
    for name, result in results.items():
        if name != 'No Regularization':
            improvement = (baseline_overfitting - result['overfitting_ratio']) / baseline_overfitting * 100
            print(f"{name}: {improvement:.2f}% reduction in overfitting")
    
    # Plot training and validation loss curves
    plt.figure(figsize=(15, 10))
    
    for i, (name, result) in enumerate(results.items()):
        plt.subplot(2, 3, i+1)
        plt.plot(result['history']['train_loss'], label='Train Loss')
        plt.plot(result['history']['val_loss'], label='Val Loss')
        plt.title(name)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot comparison of validation losses
    plt.figure(figsize=(10, 6))
    
    for name, result in results.items():
        plt.plot(result['history']['val_loss'], label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Comparison of Regularization Strategies')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("Regularization techniques testing complete!")
