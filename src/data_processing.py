class DataProcessor:
    """
    Handles data loading, preprocessing, normalization and batching for
    trajectory prediction tasks.
    """

    def __init__(self, config=None):
        """
        Initialize the data processor with configuration settings.

        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config or {}
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def load_data(self, X=None, y=None, file_path=None):
        """
        Load data either from provided arrays or from file.

        Args:
            X: Input features array (optional)
            y: Target values array (optional)
            file_path: Path to data file (optional)

        Returns:
            X, y: Loaded data
        """
        if X is not None and y is not None:
            print("Using provided data arrays")
            return X, y

        elif file_path is not None:
            print(f"Loading data from {file_path}")
            # Determine file type by extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                # Implement logic to extract X and y from dataframe
                # This depends on the specific structure of your CSV

            elif file_path.endswith('.npy'):
                data = np.load(file_path, allow_pickle=True)
                # Implement logic to extract X and y from numpy array

            elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                # Extract X and y from the loaded data

            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            # Placeholder - replace with actual extraction logic
            X = data.get('X', None)
            y = data.get('y', None)

            return X, y

        else:
            raise ValueError("Either data arrays or file path must be provided")

    def preprocess_features(self, X):
        """
        Preprocess input features.

        Args:
            X: Input features of shape (n_samples, n_timesteps, n_features)

        Returns:
            X_processed: Processed features
        """
        # Check dimensions
        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D input (samples, timesteps, features), got shape {X.shape}")

        n_samples, n_timesteps, n_features = X.shape

        # Handle missing values if any
        # For simplicity, we'll replace NaNs with 0, but more sophisticated
        # approaches like interpolation might be better for real data
        if np.isnan(X).any():
            print("Warning: NaN values detected in input features. Replacing with zeros.")
            X = np.nan_to_num(X)

        # Additional feature engineering could be added here
        # For example, computing derived features like acceleration from velocity

        return X

    def normalize_data(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        """
        Normalize input and output data using StandardScaler.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            X_test, y_test: Test data (optional)

        Returns:
            Normalized versions of all provided datasets
        """
        # Reshape to 2D for StandardScaler
        n_train_samples, n_timesteps, n_features = X_train.shape
        X_train_2d = X_train.reshape(-1, n_features)

        # Fit and transform X
        self.scaler_X.fit(X_train_2d)
        X_train_2d = self.scaler_X.transform(X_train_2d)

        # Reshape back to 3D
        X_train_normalized = X_train_2d.reshape(n_train_samples, n_timesteps, n_features)

        # For y (trajectory positions)
        n_output_features = y_train.shape[-1]  # Usually 2 for (x,y) coordinates
        y_train_2d = y_train.reshape(-1, n_output_features)

        # Fit and transform y
        self.scaler_y.fit(y_train_2d)
        y_train_2d = self.scaler_y.transform(y_train_2d)

        # Reshape back
        y_train_normalized = y_train_2d.reshape(n_train_samples, n_timesteps, n_output_features)

        # Process validation data if provided
        X_val_normalized, y_val_normalized = None, None
        if X_val is not None and y_val is not None:
            X_val_2d = X_val.reshape(-1, n_features)
            X_val_2d = self.scaler_X.transform(X_val_2d)
            X_val_normalized = X_val_2d.reshape(X_val.shape)

            y_val_2d = y_val.reshape(-1, n_output_features)
            y_val_2d = self.scaler_y.transform(y_val_2d)
            y_val_normalized = y_val_2d.reshape(y_val.shape)

        # Process test data if provided
        X_test_normalized, y_test_normalized = None, None
        if X_test is not None and y_test is not None:
            X_test_2d = X_test.reshape(-1, n_features)
            X_test_2d = self.scaler_X.transform(X_test_2d)
            X_test_normalized = X_test_2d.reshape(X_test.shape)

            y_test_2d = y_test.reshape(-1, n_output_features)
            y_test_2d = self.scaler_y.transform(y_test_2d)
            y_test_normalized = y_test_2d.reshape(y_test.shape)

        return X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, X_test_normalized, y_test_normalized

    # Add this to your DataProcessor class to fix the inverse_normalize_y method

    def inverse_normalize_y(self, y_normalized):
      """
      Transform normalized y values back to original scale.
      
      Args:
          y_normalized: Normalized y values
          
      Returns:
          y: Values in original scale
      """
      original_shape = y_normalized.shape
      n_output_features = original_shape[-1]
      
      # Check for NaN values in input
      if np.isnan(y_normalized).any():
        print("Warning: NaN values detected in input to inverse_normalize_y")
      
      # Handle the mismatch between scaler dimensions and data dimensions
      if n_output_features != self.scaler_y.scale_.shape[0]:
        # We'll only transform the first 2 dimensions (X,Y coordinates)
        # and leave the other dimensions unchanged
          
        # Reshape to 2D
        y_reshaped = y_normalized.reshape(-1, n_output_features)
          
        # Create output array
        y_transformed = y_reshaped.copy()
          
        # Only inverse transform the first 2 dimensions
        position_data = y_reshaped[:, :2]
          
        # Handle NaN values
        nan_mask = np.isnan(position_data).any(axis=1)
        if nan_mask.any():
          print(f"Warning: {np.sum(nan_mask)} rows with NaN values found in position data")
          # Replace NaNs with zeros for transformation
          position_data_clean = position_data.copy()
          position_data_clean[nan_mask] = 0
              
          # Transform the clean data
          transformed_positions = self.scaler_y.inverse_transform(position_data_clean)
              
          # Put NaNs back where they were
          transformed_positions[nan_mask] = np.nan
        else:
          transformed_positions = self.scaler_y.inverse_transform(position_data)
          
          # Put transformed positions back
          y_transformed[:, :2] = transformed_positions
          
          # Reshape back to original shape
          return y_transformed.reshape(original_shape)
      else:
          # Original code for when dimensions match
          y_normalized_2d = y_normalized.reshape(-1, n_output_features)
          
          # Handle NaN values
          nan_mask = np.isnan(y_normalized_2d).any(axis=1)
          if nan_mask.any():
              print(f"Warning: {np.sum(nan_mask)} rows with NaN values found")
              # Replace NaNs with zeros for transformation
              y_normalized_2d_clean = y_normalized_2d.copy()
              y_normalized_2d_clean[nan_mask] = 0
              
              # Transform the clean data
              y_2d = self.scaler_y.inverse_transform(y_normalized_2d_clean)
              
              # Put NaNs back where they were
              y_2d[nan_mask] = np.nan
          else:
              y_2d = self.scaler_y.inverse_transform(y_normalized_2d)
          
          return y_2d.reshape(original_shape)

    def split_data(self, X, y, val_split=0.2, test_split=0.1, shuffle=True):
        """
        Split data into training, validation, and test sets.

        Args:
            X: Input features
            y: Target values
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            shuffle: Whether to shuffle before splitting

        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test: Split datasets
        """
        # First split off test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_split, random_state=self.config.get('random_seed', 42), shuffle=shuffle
        )

        # Then split remaining data into train and validation
        val_ratio = val_split / (1 - test_split)  # Adjust validation ratio
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio,
            random_state=self.config.get('random_seed', 42), shuffle=shuffle
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def generate_batches(self, X, y, batch_size=32, shuffle=True):
        """
        Generate batches of data for training or evaluation.

        Args:
            X: Input features
            y: Target values
            batch_size: Size of each batch
            shuffle: Whether to shuffle data before generating batches

        Yields:
            X_batch, y_batch: Batches of data
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            yield X_batch, y_batch

    def prepare_data(self, X, y, val_split=None, test_split=None):
        """
        Complete data preparation pipeline.

        Args:
            X: Input features
            y: Target values
            val_split: Validation split ratio (optional)
            test_split: Test split ratio (optional)

        Returns:
            Processed and split datasets
        """
        # Use config values if not specified
        val_split = val_split or self.config.get('validation_split', 0.2)
        test_split = test_split or self.config.get('test_split', 0.1)

        # Preprocess features
        X = self.preprocess_features(X)

        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(
            X, y, val_split=val_split, test_split=test_split
        )

        # Normalize data
        X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm = self.normalize_data(
            X_train, y_train, X_val, y_val, X_test, y_test
        )

        # Store processed data
        self.X_train, self.y_train = X_train_norm, y_train_norm
        self.X_val, self.y_val = X_val_norm, y_val_norm
        self.X_test, self.y_test = X_test_norm, y_test_norm

        return (
            X_train_norm, y_train_norm,
            X_val_norm, y_val_norm,
            X_test_norm, y_test_norm
        )

# Add this to your plotting functions before creating histograms
def safe_plot_histogram(data, bins=30, **kwargs):
    """Safely plot a histogram even if data contains NaNs"""
    # Remove NaN values
    clean_data = data[~np.isnan(data)]
    if len(clean_data) == 0:
        print("Warning: All data values are NaN, cannot create histogram")
        return
    
    plt.hist(clean_data, bins=bins, **kwargs)

# Testing the data processing module with synthetic data
if __name__ == "__main__":
    # Generate synthetic data
    X, y = generate_synthetic_trajectory_data(n_samples=1000, n_timesteps=20, n_features=4)

    # Initialize data processor
    data_processor = DataProcessor(config)

    # Process the data
    X_train, y_train, X_val, y_val, X_test, y_test = data_processor.prepare_data(X, y)

    # Print shapes
    print("Training data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print("\nValidation data shapes:")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print("\nTest data shapes:")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Check normalization by comparing statistics
    print("\nVerifying normalization:")
    print(f"X_train mean: {np.mean(X_train.reshape(-1, X_train.shape[-1]), axis=0)}")
    print(f"X_train std: {np.std(X_train.reshape(-1, X_train.shape[-1]), axis=0)}")

    # Generate training batches
    print("\nGenerating batches:")
    batch_gen = data_processor.generate_batches(X_train, y_train, batch_size=32)

    # Get first batch
    X_batch, y_batch = next(batch_gen)
    print(f"Batch shapes - X: {X_batch.shape}, y: {y_batch.shape}")

    # Test inverse normalization
    y_pred_normalized = np.random.randn(*y_test.shape)  # Simulated predictions
    y_pred = data_processor.inverse_normalize_y(y_pred_normalized)
    print("\nInverse normalization test:")
    print(f"y_pred_normalized shape: {y_pred_normalized.shape}")
    print(f"y_pred (original scale) shape: {y_pred.shape}")

    print("\nData processing module testing complete!")

# Test the Data Processing Module

# Generate synthetic data or load real data
print("Generating synthetic trajectory data...")
X_data, y_data = generate_synthetic_trajectory_data(n_samples=1500, n_timesteps=20, n_features=4)

# Create data processor with our configuration
data_processor = DataProcessor(config)

# Process the data through the complete pipeline
print("\nProcessing data through the pipeline...")
X_train, y_train, X_val, y_val, X_test, y_test = data_processor.prepare_data(X_data, y_data)

# Print summary statistics
print("\n=== Dataset Summary ===")
print(f"Total samples: {len(X_data)}")
print(f"Training samples: {len(X_train)} ({len(X_train)/len(X_data)*100:.1f}%)")
print(f"Validation samples: {len(X_val)} ({len(X_val)/len(X_data)*100:.1f}%)")
print(f"Test samples: {len(X_test)} ({len(X_test)/len(X_data)*100:.1f}%)")

# Verify normalization
print("\n=== Normalization Verification ===")
X_flat = X_train.reshape(-1, X_train.shape[-1])
print(f"X_train mean: {np.mean(X_flat, axis=0)}")
print(f"X_train std: {np.std(X_flat, axis=0)}")
print(f"Expected values close to 0 for mean and 1 for std deviation")

# Visualize original vs. normalized trajectories
plt.figure(figsize=(14, 6))

# Original trajectories (sample)
plt.subplot(1, 2, 1)
for i in range(5):
    idx = np.random.randint(0, len(y_test))
    y_orig = data_processor.inverse_normalize_y(y_test[idx:idx+1])[0]
    plt.plot(y_orig[:, 0], y_orig[:, 1], 'o-', alpha=0.7, label=f'Traj {i+1}' if i==0 else "")
plt.title('Original Scale Trajectories (Sample)')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.grid(True)

# Normalized trajectories (same samples)
plt.subplot(1, 2, 2)
for i in range(5):
    idx = np.random.randint(0, len(y_test))
    plt.plot(y_test[idx, :, 0], y_test[idx, :, 1], 'o-', alpha=0.7, label=f'Traj {i+1}' if i==0 else "")
plt.title('Normalized Trajectories (Sample)')
plt.xlabel('X Position (normalized)')
plt.ylabel('Y Position (normalized)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Test batch generation
print("\n=== Testing Batch Generation ===")
batch_size = config['batch_size']
batch_gen = data_processor.generate_batches(X_train, y_train, batch_size=batch_size)
# Get and display first batch properties
X_batch, y_batch = next(batch_gen)
print(f"Batch size: {batch_size}")
print(f"X_batch shape: {X_batch.shape}")
print(f"y_batch shape: {y_batch.shape}")

# Count total batches
n_batches = 0
batch_gen = data_processor.generate_batches(X_train, y_train, batch_size=batch_size)
for _ in batch_gen:
    n_batches += 1
print(f"Total batches per epoch: {n_batches}")
print("\nData processing module test complete!")
