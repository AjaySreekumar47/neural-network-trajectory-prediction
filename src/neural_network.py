# 3. NEURAL NETWORK ARCHITECTURE
import numpy as np

class Layer:
    """Base class for neural network layers"""

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        """Forward pass through the layer"""
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        """Backward pass through the layer"""
        raise NotImplementedError

class Dense(Layer):
    """Fully connected layer"""

    def __init__(self, input_size, output_size):
        """
        Initialize a dense layer.

        Args:
            input_size: Number of input features
            output_size: Number of output features
        """
        super().__init__()
        # He initialization for weights
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((1, output_size))

        # For optimizer (will be used later)
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.bias)

        # For regularization
        self.l1_lambda = 0.0
        self.l2_lambda = 0.0

    def forward(self, input_data):
        """
        Forward pass through the dense layer.

        Args:
            input_data: Input data of shape (batch_size, input_size)

        Returns:
            Output of shape (batch_size, output_size)
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Backward pass through the dense layer.

        Args:
            output_gradient: Gradient of the loss with respect to the output
            learning_rate: Learning rate for weight updates

        Returns:
            Gradient of the loss with respect to the input
        """
        # Compute gradients
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        # Apply regularization to weights gradient
        if self.l1_lambda > 0:
            l1_grad = self.l1_lambda * np.sign(self.weights)
            weights_gradient += l1_grad

        if self.l2_lambda > 0:
            l2_grad = self.l2_lambda * self.weights
            weights_gradient += l2_grad

        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        # Compute gradient with respect to input for backpropagation
        input_gradient = np.dot(output_gradient, self.weights.T)

        return input_gradient

class Activation(Layer):
    """Base class for activation layers"""

    def __init__(self, activation, activation_prime):
        """
        Initialize an activation layer.

        Args:
            activation: Activation function
            activation_prime: Derivative of activation function
        """
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        """
        Forward pass through the activation layer.

        Args:
            input_data: Input data

        Returns:
            Activated output
        """
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Backward pass through the activation layer.

        Args:
            output_gradient: Gradient of the loss with respect to the output
            learning_rate: Learning rate (not used in activation layers)

        Returns:
            Gradient of the loss with respect to the input
        """
        return self.activation_prime(self.input) * output_gradient

# Activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))  # Clip to avoid overflow

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def linear(x):
    return x

def linear_prime(x):
    return np.ones_like(x)


# Create specific activation layer classes
class ReLU(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)

class Linear(Activation):
    def __init__(self):
        super().__init__(linear, linear_prime)

class Dropout(Layer):
    """Dropout layer for regularization"""

    def __init__(self, rate):
        """
        Initialize a dropout layer.

        Args:
            rate: Dropout rate (probability of dropping a unit)
        """
        super().__init__()
        self.rate = rate
        self.mask = None
        self.training = True

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
            # Create dropout mask
            self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape) / (1 - self.rate)
            self.output = input_data * self.mask
        else:
            # During inference, we don't drop any units
            self.output = input_data

        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Backward pass through the dropout layer.

        Args:
            output_gradient: Gradient of the loss with respect to the output
            learning_rate: Learning rate (not used in dropout layer)

        Returns:
            Gradient of the loss with respect to the input
        """
        if self.training:
            return output_gradient * self.mask
        else:
            return output_gradient

class Reshape(Layer):
    """Reshape layer to handle dimensional transformations"""

    def __init__(self, input_shape, output_shape):
        """
        Initialize a reshape layer.

        Args:
            input_shape: Shape of the input (excluding batch dimension)
            output_shape: Shape of the output (excluding batch dimension)
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input_data):
        """
        Forward pass through the reshape layer.

        Args:
            input_data: Input data

        Returns:
            Reshaped output
        """
        self.input = input_data
        batch_size = input_data.shape[0]
        self.output = input_data.reshape(batch_size, *self.output_shape)
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Backward pass through the reshape layer.

        Args:
            output_gradient: Gradient of the loss with respect to the output
            learning_rate: Learning rate (not used in reshape layer)

        Returns:
            Gradient of the loss with respect to the input
        """
        batch_size = output_gradient.shape[0]
        return output_gradient.reshape(batch_size, *self.input_shape)

class Recurrent(Layer):
    """Simple recurrent layer (RNN)"""

    def __init__(self, input_size, hidden_size):
        """
        Initialize a recurrent layer.

        Args:
            input_size: Size of input features at each time step
            hidden_size: Size of hidden state
        """
        super().__init__()

        # Input to hidden weights
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        # Hidden to hidden weights
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        # Hidden bias
        self.bh = np.zeros((1, hidden_size))

        # For backpropagation through time (BPTT)
        self.inputs = None
        self.hidden_states = None

        # For optimizer
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dbh = np.zeros_like(self.bh)

    def forward(self, input_data):
        """
        Forward pass through the recurrent layer.

        Args:
            input_data: Input sequence of shape (batch_size, sequence_length, input_size)

        Returns:
            All hidden states of shape (batch_size, sequence_length, hidden_size)
        """
        batch_size, sequence_length, _ = input_data.shape
        hidden_size = self.Whh.shape[0]

        # Store inputs for backprop
        self.inputs = input_data

        # Initialize hidden states
        h0 = np.zeros((batch_size, hidden_size))
        self.hidden_states = [h0]

        # Forward pass through time
        for t in range(sequence_length):
            xt = input_data[:, t, :]
            ht_prev = self.hidden_states[-1]

            # Update hidden state: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            ht = np.tanh(np.dot(xt, self.Wxh) + np.dot(ht_prev, self.Whh) + self.bh)
            self.hidden_states.append(ht)

        # Return all hidden states except the initial one
        self.output = np.stack(self.hidden_states[1:], axis=1)
        return self.output

    def backward(self, output_gradient, learning_rate):
        """
        Backward pass through the recurrent layer.

        Args:
            output_gradient: Gradient of the loss with respect to the output
            learning_rate: Learning rate for weight updates

        Returns:
            Gradient of the loss with respect to the input
        """
        batch_size, sequence_length, hidden_size = output_gradient.shape
        input_size = self.Wxh.shape[0]

        # Initialize gradients
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        # Initialize gradient for input
        dinputs = np.zeros((batch_size, sequence_length, input_size))

        # Initialize gradient for hidden state at t=T (final time step)
        dhnext = np.zeros((batch_size, hidden_size))

        # Backpropagate through time
        for t in reversed(range(sequence_length)):
            # Add gradient from the output
            dh = output_gradient[:, t, :] + dhnext

            # Backprop through tanh
            dtanh = (1 - self.hidden_states[t+1] ** 2) * dh

            # Update gradients
            dbh += np.sum(dtanh, axis=0, keepdims=True)
            dWxh += np.dot(self.inputs[:, t, :].T, dtanh)
            dWhh += np.dot(self.hidden_states[t].T, dtanh)

            # Compute gradient for previous hidden state and input
            dhnext = np.dot(dtanh, self.Whh.T)
            dinputs[:, t, :] = np.dot(dtanh, self.Wxh.T)

        # Clip gradients to prevent exploding gradients
        for grad in [dWxh, dWhh, dbh]:
            np.clip(grad, -5, 5, out=grad)

        # Update weights
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.bh -= learning_rate * dbh

        return dinputs

# Loss functions
class Loss:
    """Base class for loss functions"""

    def __call__(self, y_true, y_pred):
        """
        Compute the loss value.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Loss value
        """
        return np.mean(self.forward(y_true, y_pred))

    def forward(self, y_true, y_pred):
        """
        Forward pass of the loss function.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Loss values (not reduced)
        """
        raise NotImplementedError

    def backward(self, y_true, y_pred):
        """
        Compute gradient of the loss with respect to predictions.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Gradient of the loss with respect to y_pred
        """
        raise NotImplementedError

class MeanSquaredError(Loss):
    """Mean Squared Error loss function"""

    def forward(self, y_true, y_pred):
        """
        Compute MSE loss.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            MSE loss values (not reduced)
        """
        return 0.5 * (y_pred - y_true) ** 2

    def backward(self, y_true, y_pred):
        """
        Compute gradient of MSE with respect to predictions.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Gradient of MSE with respect to y_pred
        """
        return y_pred - y_true


class MeanAbsoluteError(Loss):
    """Mean Absolute Error loss function"""

    def forward(self, y_true, y_pred):
        """
        Compute MAE loss.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            MAE loss values (not reduced)
        """
        return np.abs(y_pred - y_true)

    def backward(self, y_true, y_pred):
        """
        Compute gradient of MAE with respect to predictions.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Gradient of MAE with respect to y_pred
        """
        return np.sign(y_pred - y_true)

class TrajectoryMSE(Loss):
    """Specialized MSE loss for trajectory prediction that can weight time steps differently"""

    def __init__(self, time_weights=None):
        """
        Initialize TrajectoryMSE loss.

        Args:
            time_weights: Weights for different time steps, higher weights for later steps
                         (default: None, equal weighting)
        """
        super().__init__()
        self.time_weights = time_weights

    def forward(self, y_true, y_pred):
        """
        Compute weighted MSE for trajectories.

        Args:
            y_true: True trajectories of shape (batch_size, sequence_length, 2)
            y_pred: Predicted trajectories

        Returns:
            Weighted MSE loss values
        """
        squared_errors = 0.5 * (y_pred - y_true) ** 2

        # If time weights are provided, apply them along the sequence dimension
        if self.time_weights is not None:
            # Reshape weights to match the dimensions
            weights = np.reshape(self.time_weights, (1, -1, 1))
            squared_errors = squared_errors * weights

        return np.mean(squared_errors, axis=(1, 2))

    def backward(self, y_true, y_pred):
        """
        Compute gradient of weighted MSE with respect to predictions.

        Args:
            y_true: True trajectories
            y_pred: Predicted trajectories

        Returns:
            Gradient of weighted MSE with respect to y_pred
        """
        gradient = y_pred - y_true

        # Apply time weights if provided
        if self.time_weights is not None:
            weights = np.reshape(self.time_weights, (1, -1, 1))
            gradient = gradient * weights

        # Normalize by the number of elements
        gradient = gradient / np.prod(y_true.shape)

        return gradient


# Neural Network class to combine layers
class NeuralNetwork:
    """Custom neural network with configurable layers"""

    def __init__(self):
        """Initialize an empty neural network"""
        self.layers = []
        self.loss = None
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

    def train(self, x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate,
              learning_rate_decay=0.0, l2_lambda=0.0, early_stopping_patience=None):
        """
        Train the neural network.

        Args:
            x_train: Training inputs
            y_train: Training targets
            x_val: Validation inputs
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Size of each training batch
            learning_rate: Initial learning rate
            learning_rate_decay: Learning rate decay per epoch
            l2_lambda: L2 regularization strength
            early_stopping_patience: Patience for early stopping

        Returns:
            Loss history dictionary
        """
        # Set all dropout layers to training mode
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = True
            # Apply L2 regularization to Dense layers
            if isinstance(layer, Dense):
                layer.l2_lambda = l2_lambda

        # History for tracking metrics
        history = {
            'train_loss': [],
            'val_loss': []
        }

        # For early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        # Number of training samples
        n_samples = len(x_train)

        # Training loop
        for epoch in range(epochs):
            # Decay learning rate if specified
            current_lr = learning_rate * (1.0 / (1.0 + learning_rate_decay * epoch))

            # Shuffle indices for this epoch
            indices = np.random.permutation(n_samples)

            # Track epoch loss
            epoch_loss = 0

            # Process mini-batches
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                # Get batch data
                batch_x = x_train[batch_indices]
                batch_y = y_train[batch_indices]

                # Forward pass
                output = batch_x
                for layer in self.layers:
                    output = layer.forward(output)

                # Compute loss
                batch_loss = self.loss(batch_y, output)
                epoch_loss += batch_loss * len(batch_indices)

                # Backward pass (compute initial gradient from loss)
                gradient = self.loss.backward(batch_y, output)

                # Backpropagate through layers in reverse order
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, current_lr)

            # Compute average epoch loss
            epoch_loss /= n_samples

            # Evaluate on validation set
            val_predictions = self.predict(x_val)
            val_loss = self.loss(y_val, val_predictions)

            # Save history
            history['train_loss'].append(epoch_loss)
            history['val_loss'].append(val_loss)

            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - lr: {current_lr:.6f}")

            # Check for early stopping
            if early_stopping_patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return history

    def evaluate(self, x_test, y_test):
        """
        Evaluate the network on test data.

        Args:
            x_test: Test inputs
            y_test: Test targets

        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        predictions = self.predict(x_test)

        # Compute test loss
        test_loss = self.loss(y_test, predictions)

        # Compute MSE for trajectory points
        mse_per_point = np.mean((predictions - y_test) ** 2, axis=(0, 2))

        # Compute accuracy (for a trajectory prediction task, we'll define this as
        # the percentage of predictions within a certain threshold of the true value)
        threshold = 0.1  # This is arbitrary and depends on your normalized data scale
        within_threshold = np.sqrt(np.sum((predictions - y_test) ** 2, axis=2)) < threshold
        accuracy = np.mean(within_threshold) * 100

        # Return metrics
        return {
            'test_loss': test_loss,
            'mse_per_timestep': mse_per_point,
            'accuracy': accuracy
        }

# Example: Creating a TrajectoryNN model
def create_trajectory_prediction_model(input_shape, output_shape):
    """
    Create a neural network for trajectory prediction.

    Args:
        input_shape: Shape of input data (timesteps, features)
        output_shape: Shape of output data (timesteps, output_dims)

    Returns:
        Configured neural network
    """
    model = NeuralNetwork()

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

    return model

# Testing the neural network implementation
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

    # Create the model
    model = create_trajectory_prediction_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_shape=(y_train.shape[1], y_train.shape[2])
    )

    # Train the model with a small number of epochs for testing
    history = model.train(
        x_train=X_train,
        y_train=y_train,
        x_val=X_val,
        y_val=y_val,
        epochs=5,  # Small number for testing
        batch_size=16,
        learning_rate=0.01,
        learning_rate_decay=0.1,
        l2_lambda=0.001,
        early_stopping_patience=3
    )

    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    print("\nTest metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Plot the training history
    plt.figure(figsize=(10, 4))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Generate predictions on test data
    predictions = model.predict(X_test)

    # Transform back to original scale
    y_test_orig = data_processor.inverse_normalize_y(y_test)
    pred_orig = data_processor.inverse_normalize_y(predictions)

    # Plot a few test trajectories vs predictions
    plt.figure(figsize=(12, 8))
    for i in range(3):  # Plot 3 examples
        plt.subplot(1, 3, i+1)

        # Plot true trajectory
        plt.plot(y_test_orig[i, :, 0], y_test_orig[i, :, 1], 'b-o', label='True')

        # Plot predicted trajectory
        plt.plot(pred_orig[i, :, 0], pred_orig[i, :, 1], 'r-x', label='Predicted')

        plt.grid(True)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Trajectory {i+1}')
        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.show()

    print("Neural network testing complete!")

## Testing Neural Network Implementation

# Test our Neural Network Implementation

print("Starting neural network testing...")

# Parameters for a more thorough test
np.random.seed(42)
n_samples = 1000
n_timesteps = 20
n_features = 4

print(f"Generating {n_samples} synthetic trajectory samples...")
# Generate synthetic trajectory data
X, y = generate_synthetic_trajectory_data(
    n_samples=n_samples,
    n_timesteps=n_timesteps,
    n_features=n_features
)

# Process the data
print("Processing data...")
data_processor = DataProcessor(config)
X_train, y_train, X_val, y_val, X_test, y_test = data_processor.prepare_data(
    X, y, val_split=0.2, test_split=0.1
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Create the model
print("Creating neural network model...")
model = create_trajectory_prediction_model(
    input_shape=(X_train.shape[1], X_train.shape[2]),
    output_shape=(y_train.shape[1], y_train.shape[2])
)

# Define training configuration
training_config = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.005,
    'learning_rate_decay': 0.05,
    'l2_lambda': 0.001,
    'early_stopping_patience': 5
}

print("\nTraining configuration:")
for key, value in training_config.items():
    print(f"  {key}: {value}")

# Train the model
print("\nStarting model training...")
start_time = time.time()

history = model.train(
    x_train=X_train,
    y_train=y_train,
    x_val=X_val,
    y_val=y_val,
    epochs=training_config['epochs'],
    batch_size=training_config['batch_size'],
    learning_rate=training_config['learning_rate'],
    learning_rate_decay=training_config['learning_rate_decay'],
    l2_lambda=training_config['l2_lambda'],
    early_stopping_patience=training_config['early_stopping_patience']
)

# Calculate training time
training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# Evaluate the model on test data
print("\nEvaluating model on test data...")
metrics = model.evaluate(X_test, y_test)
print("\nTest metrics:")
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Generate predictions on test data
print("\nGenerating predictions on test data...")
predictions = model.predict(X_test)

# Calculate accuracy based on distance threshold
thresholds = [0.05, 0.1, 0.2, 0.5]
print("\nAccuracy at different distance thresholds:")
for threshold in thresholds:
    distances = np.sqrt(np.sum((predictions - y_test) ** 2, axis=2))
    within_threshold = distances < threshold
    accuracy_per_timestep = np.mean(within_threshold, axis=0) * 100
    overall_accuracy = np.mean(within_threshold) * 100

    print(f"  Threshold {threshold:.2f}:")
    print(f"    Overall accuracy: {overall_accuracy:.2f}%")
    print(f"    Final position accuracy: {accuracy_per_timestep[-1]:.2f}%")

# Transform predictions back to original scale for visualization
y_test_orig = data_processor.inverse_normalize_y(y_test)
pred_orig = data_processor.inverse_normalize_y(predictions)

# Plot trajectories for specific examples
num_samples = min(4, len(y_test))
plt.figure(figsize=(15, 10))

for i in range(num_samples):
    plt.subplot(2, 2, i+1)

    # Plot true trajectory
    plt.plot(y_test_orig[i, :, 0], y_test_orig[i, :, 1], 'b-o',
             label='True', markersize=4, alpha=0.7)

    # Plot predicted trajectory
    plt.plot(pred_orig[i, :, 0], pred_orig[i, :, 1], 'r-x',
             label='Predicted', markersize=4, alpha=0.7)

    # Mark start and end points
    plt.scatter(y_test_orig[i, 0, 0], y_test_orig[i, 0, 1],
                color='green', s=100, marker='o', label='Start')
    plt.scatter(y_test_orig[i, -1, 0], y_test_orig[i, -1, 1],
                color='blue', s=100, marker='s', label='True End')
    plt.scatter(pred_orig[i, -1, 0], pred_orig[i, -1, 1],
                color='red', s=100, marker='x', label='Predicted End')

    plt.grid(True)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Test Trajectory {i+1}')
    if i == 0:
        plt.legend(loc='best')

plt.tight_layout()
plt.show()

# Calculate error distribution
errors = np.sqrt(np.sum((predictions - y_test) ** 2, axis=2))
plt.figure(figsize=(12, 5))

# Plot 1: Error by time step
plt.subplot(1, 2, 1)
mean_errors = np.mean(errors, axis=0)
std_errors = np.std(errors, axis=0)
time_steps = np.arange(len(mean_errors))

plt.plot(time_steps, mean_errors, 'b-', label='Mean Error')
plt.fill_between(time_steps, mean_errors - std_errors, mean_errors + std_errors,
                 alpha=0.3, label='±1 Std Dev')
plt.xlabel('Time Step')
plt.ylabel('Error (Euclidean Distance)')
plt.title('Error by Time Step')
plt.grid(True)
plt.legend()

# Plot 2: Overall error distribution
plt.subplot(1, 2, 2)
plt.hist(errors.flatten(), bins=30, alpha=0.7, color='blue')
plt.axvline(np.mean(errors), color='red', linestyle='dashed',
            linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
plt.xlabel('Error (Euclidean Distance)')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Save the model (simple pickle implementation)
print("\nSaving model...")
model_path = os.path.join(config['models_path'], 'trajectory_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_path}")

print("\nNeural network testing completed!")
