# 6. EVALUATION FRAMEWORK

class EvaluationMetrics:
    """Class for computing evaluation metrics for trajectory prediction"""
    
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        Compute Mean Squared Error.
        
        Args:
            y_true: True trajectories of shape (n_samples, n_timesteps, n_dims)
            y_pred: Predicted trajectories
            
        Returns:
            MSE value
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        """
        Compute Mean Absolute Error.
        
        Args:
            y_true: True trajectories
            y_pred: Predicted trajectories
            
        Returns:
            MAE value
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        """
        Compute Root Mean Squared Error.
        
        Args:
            y_true: True trajectories
            y_pred: Predicted trajectories
            
        Returns:
            RMSE value
        """
        return np.sqrt(EvaluationMetrics.mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def trajectory_distance_error(y_true, y_pred):
        """
        Compute average Euclidean distance between predicted and true trajectories.
        
        Args:
            y_true: True trajectories of shape (n_samples, n_timesteps, 2)
            y_pred: Predicted trajectories
            
        Returns:
            Average distance error
        """
        # Compute Euclidean distance at each time step
        distances = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=2))
        
        # Average across samples and time steps
        return np.mean(distances)
    
    @staticmethod
    def final_displacement_error(y_true, y_pred):
        """
        Compute displacement error at final time step.
        
        Args:
            y_true: True trajectories
            y_pred: Predicted trajectories
            
        Returns:
            Final displacement error
        """
        # Extract final positions
        final_true = y_true[:, -1, :]
        final_pred = y_pred[:, -1, :]
        
        # Compute Euclidean distance at final step
        distances = np.sqrt(np.sum((final_true - final_pred) ** 2, axis=1))
        
        # Average across samples
        return np.mean(distances)
    
    @staticmethod
    def position_accuracy(y_true, y_pred, threshold=0.5):
        """
        Compute position accuracy within a threshold.
        
        Args:
            y_true: True trajectories
            y_pred: Predicted trajectories
            threshold: Distance threshold for considering a position as correctly predicted
            
        Returns:
            Accuracy between 0 and 1
        """
        # Compute Euclidean distance at each time step
        distances = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=2))
        
        # Count positions within threshold
        within_threshold = distances < threshold
        
        # Compute accuracy
        return np.mean(within_threshold)
    
    @staticmethod
    def compute_all_metrics(y_true, y_pred, thresholds=None):
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: True trajectories
            y_pred: Predicted trajectories
            thresholds: List of distance thresholds for accuracy metrics
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Default thresholds if not provided
        if thresholds is None:
            thresholds = [0.1, 0.5, 1.0]
        
        # Compute basic metrics
        metrics['mse'] = EvaluationMetrics.mean_squared_error(y_true, y_pred)
        metrics['mae'] = EvaluationMetrics.mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = EvaluationMetrics.root_mean_squared_error(y_true, y_pred)
        metrics['tde'] = EvaluationMetrics.trajectory_distance_error(y_true, y_pred)
        metrics['fde'] = EvaluationMetrics.final_displacement_error(y_true, y_pred)
        
        # Compute accuracy at different thresholds
        for threshold in thresholds:
            key = f'accuracy_{threshold}'
            metrics[key] = EvaluationMetrics.position_accuracy(y_true, y_pred, threshold)
        
        return metrics


class TrajectoryVisualizer:
    """Class for visualizing trajectory predictions"""
    
    @staticmethod
    def plot_trajectory(true_trajectory, pred_trajectory=None, title=None, figsize=(10, 8)):
        """
        Plot a single trajectory.
        
        Args:
            true_trajectory: True trajectory of shape (n_timesteps, 2)
            pred_trajectory: Predicted trajectory (optional)
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot true trajectory
        plt.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'b-o', 
                 label='True Trajectory', markersize=4, alpha=0.7)
        
        if pred_trajectory is not None:
            # Plot predicted trajectory
            plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r-x',
                     label='Predicted Trajectory', markersize=4, alpha=0.7)
            
            # Mark start and end points
            plt.scatter(true_trajectory[0, 0], true_trajectory[0, 1], 
                        color='green', s=100, marker='o', label='Start')
            plt.scatter(true_trajectory[-1, 0], true_trajectory[-1, 1], 
                        color='blue', s=100, marker='s', label='True End')
            plt.scatter(pred_trajectory[-1, 0], pred_trajectory[-1, 1], 
                        color='red', s=100, marker='x', label='Predicted End')
        else:
            # Mark start and end points
            plt.scatter(true_trajectory[0, 0], true_trajectory[0, 1], 
                        color='green', s=100, marker='o', label='Start')
            plt.scatter(true_trajectory[-1, 0], true_trajectory[-1, 1], 
                        color='blue', s=100, marker='s', label='End')
        
        plt.grid(True)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(title or 'Trajectory')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_multiple_trajectories(true_trajectories, pred_trajectories=None, 
                                  indices=None, n_plots=4, figsize=(15, 10)):
        """
        Plot multiple trajectories.
        
        Args:
            true_trajectories: True trajectories of shape (n_samples, n_timesteps, 2)
            pred_trajectories: Predicted trajectories (optional)
            indices: Indices of trajectories to plot (optional)
            n_plots: Number of trajectories to plot
            figsize: Figure size
        """
        if indices is None:
            if len(true_trajectories) <= n_plots:
                indices = range(len(true_trajectories))
            else:
                indices = np.random.choice(len(true_trajectories), n_plots, replace=False)
        
        # Create subplot grid
        n_rows = int(np.ceil(len(indices) / 2))
        n_cols = min(len(indices), 2)
        
        plt.figure(figsize=figsize)
        
        for i, idx in enumerate(indices):
            plt.subplot(n_rows, n_cols, i+1)
            
            # Get true trajectory
            true_traj = true_trajectories[idx]
            
            # Get predicted trajectory if available
            pred_traj = None
            if pred_trajectories is not None:
                pred_traj = pred_trajectories[idx]
            
            # Plot trajectories
            plt.plot(true_traj[:, 0], true_traj[:, 1], 'b-o', 
                     label='True', markersize=4, alpha=0.7)
            
            if pred_traj is not None:
                plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'r-x',
                         label='Predicted', markersize=4, alpha=0.7)
                
                # Mark start and end points
                plt.scatter(true_traj[0, 0], true_traj[0, 1], 
                            color='green', s=100, marker='o', label='Start')
                plt.scatter(true_traj[-1, 0], true_traj[-1, 1], 
                            color='blue', s=100, marker='s', label='True End')
                plt.scatter(pred_traj[-1, 0], pred_traj[-1, 1], 
                            color='red', s=100, marker='x', label='Pred End')
            else:
                # Mark start and end points
                plt.scatter(true_traj[0, 0], true_traj[0, 1], 
                            color='green', s=100, marker='o', label='Start')
                plt.scatter(true_traj[-1, 0], true_traj[-1, 1], 
                            color='blue', s=100, marker='s', label='End')
            
            plt.grid(True)
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title(f'Trajectory {idx}')
            
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_error_by_timestep(true_trajectories, pred_trajectories, figsize=(12, 6)):
        """
        Plot error as a function of time step.
        
        Args:
            true_trajectories: True trajectories
            pred_trajectories: Predicted trajectories
            figsize: Figure size
        """
        # Compute Euclidean distance at each time step
        distances = np.sqrt(np.sum((true_trajectories - pred_trajectories) ** 2, axis=2))
        
        # Compute mean and std of errors at each time step
        mean_errors = np.mean(distances, axis=0)
        std_errors = np.std(distances, axis=0)
        
        # Generate time steps
        time_steps = np.arange(len(mean_errors))
        
        plt.figure(figsize=figsize)
        
        # Plot mean error with error bars
        plt.plot(time_steps, mean_errors, 'b-', label='Mean Error')
        plt.fill_between(time_steps, 
                         mean_errors - std_errors, 
                         mean_errors + std_errors, 
                         alpha=0.3, label='±1 Std Dev')
        
        plt.grid(True)
        plt.xlabel('Time Step')
        plt.ylabel('Error (Euclidean Distance)')
        plt.title('Prediction Error by Time Step')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_error_distribution(true_trajectories, pred_trajectories, figsize=(12, 5)):
        """
        Plot distribution of prediction errors.
        
        Args:
            true_trajectories: True trajectories
            pred_trajectories: Predicted trajectories
            figsize: Figure size
        """
        # Compute Euclidean distance at each time step
        distances = np.sqrt(np.sum((true_trajectories - pred_trajectories) ** 2, axis=2))
        
        plt.figure(figsize=figsize)
        
        # Plot 1: Overall error distribution
        plt.subplot(1, 2, 1)
        plt.hist(distances.flatten(), bins=30, alpha=0.7, color='blue')
        plt.axvline(np.mean(distances), color='red', linestyle='dashed', 
                    linewidth=2, label=f'Mean: {np.mean(distances):.4f}')
        plt.axvline(np.median(distances), color='green', linestyle='dashed', 
                    linewidth=2, label=f'Median: {np.median(distances):.4f}')
        plt.xlabel('Error (Euclidean Distance)')
        plt.ylabel('Frequency')
        plt.title('Overall Error Distribution')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Final position error distribution
        plt.subplot(1, 2, 2)
        final_distances = np.sqrt(np.sum((true_trajectories[:, -1, :] - pred_trajectories[:, -1, :]) ** 2, axis=1))
        plt.hist(final_distances, bins=20, alpha=0.7, color='orange')
        plt.axvline(np.mean(final_distances), color='red', linestyle='dashed', 
                    linewidth=2, label=f'Mean: {np.mean(final_distances):.4f}')
        plt.axvline(np.median(final_distances), color='green', linestyle='dashed', 
                    linewidth=2, label=f'Median: {np.median(final_distances):.4f}')
        plt.xlabel('Final Position Error')
        plt.ylabel('Frequency')
        plt.title('Final Position Error Distribution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_training_history(history, figsize=(12, 5)):
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot 1: Training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], 'b-', label='Training Loss')
        plt.plot(history['val_loss'], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Learning rate if available
        if 'learning_rate' in history:
            plt.subplot(1, 2, 2)
            plt.plot(history['learning_rate'], 'g-')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()


class ModelComparison:
    """Class for comparing multiple models or configurations"""
    
    def __init__(self):
        """Initialize model comparison"""
        self.models = {}
        self.results = {}
    
    def add_model(self, name, model, config=None):
        """
        Add a model for comparison.
        
        Args:
            name: Model name
            model: Model instance
            config: Model configuration (optional)
        """
        self.models[name] = {
            'model': model,
            'config': config
        }
    
    def evaluate_all(self, X_test, y_test, data_processor=None):
        """
        Evaluate all models.
        
        Args:
            X_test: Test inputs
            y_test: Test targets
            data_processor: Data processor for denormalization (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        for name, model_info in self.models.items():
            print(f"Evaluating model: {name}")
            
            model = model_info['model']
            
            # Generate predictions
            y_pred = model.predict(X_test)
            
            # Denormalize if data processor is provided
            if data_processor is not None:
                y_test_orig = data_processor.inverse_normalize_y(y_test)
                y_pred_orig = data_processor.inverse_normalize_y(y_pred)
            else:
                y_test_orig = y_test
                y_pred_orig = y_pred
            
            # Compute metrics
            metrics = EvaluationMetrics.compute_all_metrics(y_test_orig, y_pred_orig)
            
            # Store results
            self.results[name] = {
                'metrics': metrics,
                'predictions': y_pred_orig,
                'true_values': y_test_orig
            }
        
        return self.results
    
    def print_comparison_table(self):
        """Print comparison table of model metrics"""
        if not self.results:
            print("No evaluation results available. Run evaluate_all() first.")
            return
        
        # Collect all metric names
        all_metrics = set()
        for name, result in self.results.items():
            all_metrics.update(result['metrics'].keys())
        
        # Create and print the table
        print("\n=== Model Comparison ===")
        
        # Print header
        header = "Metric".ljust(20)
        for name in self.results.keys():
            header += f" | {name}".ljust(15)
        print(header)
        print("-" * len(header))
        
        # Print metrics
        for metric in sorted(all_metrics):
            row = metric.ljust(20)
            for name, result in self.results.items():
                value = result['metrics'].get(metric, "N/A")
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                row += f" | {value_str}".ljust(15)
            print(row)
    
    def plot_comparative_bar_chart(self, metrics=None):
        """
        Plot bar chart comparing key metrics across models.
        
        Args:
            metrics: List of metrics to compare (optional)
        """
        if not self.results:
            print("No evaluation results available. Run evaluate_all() first.")
            return
        
        # Default metrics to compare
        if metrics is None:
            metrics = ['rmse', 'fde', 'accuracy_0.5']
        
        # Get model names and extract metric values
        model_names = list(self.results.keys())
        metric_values = {}
        
        for metric in metrics:
            values = []
            for name in model_names:
                if metric in self.results[name]['metrics']:
                    values.append(self.results[name]['metrics'][metric])
                else:
                    values.append(0)
            metric_values[metric] = values
        
        # Create bar chart
        n_metrics = len(metrics)
        n_models = len(model_names)
        
        plt.figure(figsize=(12, 6))
        bar_width = 0.8 / n_metrics
        
        for i, metric in enumerate(metrics):
            x = np.arange(n_models) + i * bar_width
            plt.bar(x, metric_values[metric], width=bar_width, label=metric)
        
        plt.xlabel('Model')
        plt.ylabel('Value')
        plt.title('Model Comparison by Metrics')
        plt.xticks(np.arange(n_models) + (n_metrics-1) * bar_width / 2, model_names)
        plt.legend()
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_best_and_worst_predictions(self, model_name, n_samples=3):
        """
        Visualize best and worst predictions for a specific model.
        
        Args:
            model_name: Name of the model to visualize
            n_samples: Number of samples to visualize
        """
        if model_name not in self.results:
            print(f"Model {model_name} not found in results.")
            return
        
        result = self.results[model_name]
        true_values = result['true_values']
        predictions = result['predictions']
        
        # Compute errors for each sample
        errors = np.mean(np.sqrt(np.sum((true_values - predictions) ** 2, axis=2)), axis=1)
        
        # Get indices of best and worst predictions
        best_indices = np.argsort(errors)[:n_samples]
        worst_indices = np.argsort(errors)[-n_samples:]
        
        # Visualize best predictions
        print(f"\nBest predictions for model: {model_name}")
        TrajectoryVisualizer.plot_multiple_trajectories(
            true_values, predictions, indices=best_indices, n_plots=n_samples
        )
        
        # Visualize worst predictions
        print(f"\nWorst predictions for model: {model_name}")
        TrajectoryVisualizer.plot_multiple_trajectories(
            true_values, predictions, indices=worst_indices, n_plots=n_samples
        )


class BaselineModel:
    """Simple baseline models for trajectory prediction"""
    
    @staticmethod
    def constant_velocity(x_train, y_train, x_test):
        """
        Predict trajectory assuming constant velocity.
        
        Args:
            x_train: Training inputs
            y_train: Training trajectories
            x_test: Test inputs
            
        Returns:
            Predicted trajectories
        """
        # Get shapes
        n_test_samples = x_test.shape[0]
        n_timesteps = y_train.shape[1]
        
        # Create output array
        y_pred = np.zeros((n_test_samples, n_timesteps, 2))
        
        # For each test sample
        for i in range(n_test_samples):
            # Initialize with starting position (assuming first position is available)
            y_pred[i, 0, :] = x_test[i, 0, :2]  # Use first two features as initial position
            
            # Estimate velocity from initial positions in training data
            avg_velocity = np.zeros(2)
            count = 0
            
            for j in range(len(y_train)):
                if np.all(abs(y_train[j, 0, :] - y_pred[i, 0, :]) < 1.0):  # Similar starting position
                    # Calculate velocity from first few points
                    if n_timesteps > 3:
                        velocity = (y_train[j, 3, :] - y_train[j, 0, :]) / 3
                    else:
                        velocity = (y_train[j, 1, :] - y_train[j, 0, :])
                    
                    avg_velocity += velocity
                    count += 1
            
            # Use average velocity, or zero if no similar trajectories found
            if count > 0:
                avg_velocity /= count
            
            # Predict future positions using constant velocity
            for t in range(1, n_timesteps):
                y_pred[i, t, :] = y_pred[i, 0, :] + t * avg_velocity
        
        return y_pred
    
    @staticmethod
    def linear_extrapolation(x_train, y_train, x_test):
        """
        Predict trajectory using linear extrapolation.
        
        Args:
            x_train: Training inputs
            y_train: Training trajectories
            x_test: Test inputs
            
        Returns:
            Predicted trajectories
        """
        # Get shapes
        n_test_samples = x_test.shape[0]
        n_timesteps = y_train.shape[1]
        
        # Create output array
        y_pred = np.zeros((n_test_samples, n_timesteps, 2))
        
        # For each test sample
        for i in range(n_test_samples):
            # Use the first two points for linear extrapolation
            # Assuming we have at least 2 initial points in the test sequence
            if x_test.shape[1] >= 2:
                initial_pos = x_test[i, 0, :2]
                next_pos = x_test[i, 1, :2]
                
                # Calculate direction vector
                direction = next_pos - initial_pos
                
                # Extrapolate future positions
                for t in range(n_timesteps):
                    y_pred[i, t, :] = initial_pos + t * direction
            else:
                # Fall back to constant velocity if not enough points
                y_pred[i] = BaselineModel.constant_velocity(x_train, y_train, x_test[i:i+1])[0]
        
        return y_pred
    
    @staticmethod
    def nearest_neighbor(x_train, y_train, x_test, k=3):
        """
        Predict trajectory by finding similar trajectories in training data.
        
        Args:
            x_train: Training inputs
            y_train: Training trajectories
            x_test: Test inputs
            k: Number of nearest neighbors to consider
            
        Returns:
            Predicted trajectories
        """
        # Get shapes
        n_test_samples = x_test.shape[0]
        n_train_samples = x_train.shape[0]
        n_timesteps = y_train.shape[1]
        
        # Create output array
        y_pred = np.zeros((n_test_samples, n_timesteps, 2))
        
        # For each test sample
        for i in range(n_test_samples):
            # Get initial position
            initial_pos = x_test[i, 0, :2]
            
            # Calculate distances to all training samples
            distances = np.zeros(n_train_samples)
            for j in range(n_train_samples):
                distances[j] = np.sqrt(np.sum((y_train[j, 0, :] - initial_pos) ** 2))
            
            # Get indices of k nearest neighbors
            neighbor_indices = np.argsort(distances)[:k]
            
            # Average the trajectories of nearest neighbors
            y_pred[i] = np.mean(y_train[neighbor_indices], axis=0)
        
        return y_pred


# Testing the evaluation framework with baseline models
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_timesteps = 15
    n_features = 4
    
    print(f"Generating {n_samples} synthetic trajectory samples...")
    X, y = generate_synthetic_trajectory_data(
        n_samples=n_samples, 
        n_timesteps=n_timesteps, 
        n_features=n_features
    )
    
    # Process the data
    print("Processing data...")
    data_processor = DataProcessor(config)
    X_train, y_train, X_val, y_val, X_test, y_test = data_processor.prepare_data(
        X, y, val_split=0.2, test_split=0.2
    )
    
    # Create neural network model
    print("Creating neural network model...")
    
    # Create model with regularization
    nn_model = create_regularized_trajectory_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_shape=(y_train.shape[1], y_train.shape[2]),
        regularization_config={
            'l2': 0.001,
            'dropout_rate': 0.2,
            'batch_norm': True
        }
    )
    
    # Set optimizer (Adam)
    optimizer = Adam(learning_rate=0.001)
    nn_model.set_optimizer(optimizer)
    
    # Train the neural network (with shorter training for testing)
    print("\nTraining neural network model...")
    history = nn_model.train(
        x_train=X_train, 
        y_train=y_train,
        x_val=X_val,
        y_val=y_val,
        epochs=20,  # Reduced for testing
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
    )
    
    # Create baseline models
    print("\nCreating baseline models...")
    
    # Initialize model comparison
    model_comparison = ModelComparison()
    
    # Add neural network model
    model_comparison.add_model('Neural Network', nn_model)
    
    # Add baseline models by making predictions directly
    
    # Constant Velocity baseline
    print("Computing Constant Velocity baseline predictions...")
    cv_predictions = BaselineModel.constant_velocity(X_train, y_train, X_test)
    
    # Linear Extrapolation baseline
    print("Computing Linear Extrapolation baseline predictions...")
    le_predictions = BaselineModel.linear_extrapolation(X_train, y_train, X_test)
    
    # Nearest Neighbor baseline
    print("Computing Nearest Neighbor baseline predictions...")
    nn_predictions = BaselineModel.nearest_neighbor(X_train, y_train, X_test, k=3)
    
    # Get Neural Network predictions
    nn_predictions_norm = nn_model.predict(X_test)
    
    # Denormalize predictions for evaluation
    print("\nDenormalizing predictions for evaluation...")
    y_test_orig = data_processor.inverse_normalize_y(y_test)
    cv_predictions_orig = data_processor.inverse_normalize_y(cv_predictions)
    le_predictions_orig = data_processor.inverse_normalize_y(le_predictions)
    nn_baseline_predictions_orig = data_processor.inverse_normalize_y(nn_predictions)
    nn_predictions_orig = data_processor.inverse_normalize_y(nn_predictions_norm)
    
    # Compute metrics for each model/baseline
    print("\nComputing metrics for all models...")
    
    # Neural Network metrics
    nn_metrics = EvaluationMetrics.compute_all_metrics(y_test_orig, nn_predictions_orig)
    print("\nNeural Network Model Metrics:")
    for key, value in nn_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Constant Velocity baseline metrics
    cv_metrics = EvaluationMetrics.compute_all_metrics(y_test_orig, cv_predictions_orig)
    print("\nConstant Velocity Baseline Metrics:")
    for key, value in cv_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Linear Extrapolation baseline metrics
    le_metrics = EvaluationMetrics.compute_all_metrics(y_test_orig, le_predictions_orig)
    print("\nLinear Extrapolation Baseline Metrics:")
    for key, value in le_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Nearest Neighbor baseline metrics
    nn_baseline_metrics = EvaluationMetrics.compute_all_metrics(y_test_orig, nn_baseline_predictions_orig)
    print("\nNearest Neighbor Baseline Metrics:")
    for key, value in nn_baseline_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Calculate improvement over baselines
    print("\nNeural Network improvement over baselines:")
    
    # vs Constant Velocity
    rmse_improvement_cv = (cv_metrics['rmse'] - nn_metrics['rmse']) / cv_metrics['rmse'] * 100
    print(f"  RMSE improvement vs Constant Velocity: {rmse_improvement_cv:.2f}%")
    
    # vs Linear Extrapolation
    rmse_improvement_le = (le_metrics['rmse'] - nn_metrics['rmse']) / le_metrics['rmse'] * 100
    print(f"  RMSE improvement vs Linear Extrapolation: {rmse_improvement_le:.2f}%")
    
    # vs Nearest Neighbor
    rmse_improvement_nn = (nn_baseline_metrics['rmse'] - nn_metrics['rmse']) / nn_baseline_metrics['rmse'] * 100
    print(f"  RMSE improvement vs Nearest Neighbor: {rmse_improvement_nn:.2f}%")
    
    # Visualize trajectories
    print("\nVisualizing trajectory predictions...")
    
    # Sample trajectories for visualization
    sample_indices = np.random.choice(len(y_test), 4, replace=False)
    
    # Visualize neural network predictions
    print("\nNeural Network predictions:")
    TrajectoryVisualizer.plot_multiple_trajectories(
        y_test_orig, nn_predictions_orig, indices=sample_indices
    )
    
    # Visualize baseline predictions
    print("\nBaseline model predictions (for the first sample):")
    plt.figure(figsize=(15, 5))
    
    # Get the first sample
    sample_idx = sample_indices[0]
    
    # True trajectory
    plt.subplot(1, 3, 1)
    plt.plot(y_test_orig[sample_idx, :, 0], y_test_orig[sample_idx, :, 1], 'b-o', 
             label='True', markersize=4, alpha=0.7)
    plt.plot(cv_predictions_orig[sample_idx, :, 0], cv_predictions_orig[sample_idx, :, 1], 'r-x',
             label='Constant Velocity', markersize=4, alpha=0.7)
    plt.grid(True)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Constant Velocity Baseline')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(y_test_orig[sample_idx, :, 0], y_test_orig[sample_idx, :, 1], 'b-o', 
             label='True', markersize=4, alpha=0.7)
    plt.plot(le_predictions_orig[sample_idx, :, 0], le_predictions_orig[sample_idx, :, 1], 'g-x',
             label='Linear Extrapolation', markersize=4, alpha=0.7)
    plt.grid(True)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Linear Extrapolation Baseline')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(y_test_orig[sample_idx, :, 0], y_test_orig[sample_idx, :, 1], 'b-o', 
             label='True', markersize=4, alpha=0.7)
    plt.plot(nn_baseline_predictions_orig[sample_idx, :, 0], nn_baseline_predictions_orig[sample_idx, :, 1], 'm-x',
             label='Nearest Neighbor', markersize=4, alpha=0.7)
    plt.grid(True)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Nearest Neighbor Baseline')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot error by time step for all models
    print("\nComparing prediction error by time step:")
    plt.figure(figsize=(10, 6))
    
    # Compute errors by time step for each model
    nn_distances = np.sqrt(np.sum((y_test_orig - nn_predictions_orig) ** 2, axis=2))
    cv_distances = np.sqrt(np.sum((y_test_orig - cv_predictions_orig) ** 2, axis=2))
    le_distances = np.sqrt(np.sum((y_test_orig - le_predictions_orig) ** 2, axis=2))
    nn_baseline_distances = np.sqrt(np.sum((y_test_orig - nn_baseline_predictions_orig) ** 2, axis=2))
    
    # Compute mean errors at each time step
    nn_mean_errors = np.mean(nn_distances, axis=0)
    cv_mean_errors = np.mean(cv_distances, axis=0)
    le_mean_errors = np.mean(le_distances, axis=0)
    nn_baseline_mean_errors = np.mean(nn_baseline_distances, axis=0)
    
    # Generate time steps
    time_steps = np.arange(len(nn_mean_errors))
    
    # Plot mean errors
    plt.plot(time_steps, nn_mean_errors, 'b-', label='Neural Network')
    plt.plot(time_steps, cv_mean_errors, 'r-', label='Constant Velocity')
    plt.plot(time_steps, le_mean_errors, 'g-', label='Linear Extrapolation')
    plt.plot(time_steps, nn_baseline_mean_errors, 'm-', label='Nearest Neighbor')
    
    plt.grid(True)
    plt.xlabel('Time Step')
    plt.ylabel('Error (Euclidean Distance)')
    plt.title('Prediction Error by Time Step')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot training history
    TrajectoryVisualizer.plot_training_history(history)
    
    print("\nEvaluation framework testing complete!")
