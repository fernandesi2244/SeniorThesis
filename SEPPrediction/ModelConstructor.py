from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
import PhotosphericDataLoader
import CoronalDataLoader
import NumericDataLoader

class ModelConstructor(object):
    RANDOM_STATE = 42

    # TODO: Add another mode where you can disable PCA. This will
    # let us test more complex models that are even more interpretable
    # in some sense because we can perform convolution just on certain
    # parts of the input, for example.
    @classmethod
    def create_model(cls, dataloader_type, model_type, granularity, n_components, **kwargs):
        """
        Possible model types:

        'random_forest_simple',
        'random_forest_complex',
        'isolation_forest',
        'gaussian_RBF',
        'gaussian_matern',
        'nn_simple',
        'nn_complex',
        'logistic_regression_v1',
        'logistic_regression_v2',
        'gbm',
        'lightgbm',
        'xgboost',
        'svm_rbf',
        'svm_poly',
        'knn_v1',
        'knn_v2',
        'knn_v3',
        """

        if dataloader_type == 'photospheric':
            dataloader = PhotosphericDataLoader.SEPInputDataGenerator
        elif dataloader_type == 'coronal':
            dataloader = CoronalDataLoader.SEPInputDataGenerator
        elif dataloader_type == 'numeric':
            dataloader = NumericDataLoader.SEPInputDataGenerator
        else:
            raise ValueError('Invalid dataloader type')

        # Dictionary mapping model types to their corresponding functions
        model_map = {
            'random_forest_simple': lambda: ModelConstructor.get_rf_model(n_components, 1),
            'random_forest_complex': lambda: ModelConstructor.get_rf_model(n_components, 2),
            'isolation_forest': lambda: ModelConstructor.get_if_model(n_components, kwargs.get('contamination')),
            'gaussian_RBF': lambda: ModelConstructor.get_gaussian_model(n_components, 'RBF'),
            'gaussian_matern': lambda: ModelConstructor.get_gaussian_model(n_components, 'matern'),
            'logistic_regression_v1': lambda: ModelConstructor.get_logistic_regression_model(n_components, 1),
            'logistic_regression_v2': lambda: ModelConstructor.get_logistic_regression_model(n_components, 2),
            'gbm': lambda: ModelConstructor.get_gbm_model(n_components),
            'lightgbm': lambda: ModelConstructor.get_lightgbm_model(n_components),
            'xgboost': lambda: ModelConstructor.get_xgboost_model(n_components),
            'svm_rbf': lambda: ModelConstructor.get_svm_model(n_components, 'rbf'),
            'svm_poly': lambda: ModelConstructor.get_svm_model(n_components, 'poly'),
            'knn_v1': lambda: ModelConstructor.get_knn_model(n_components, 1),
            'knn_v2': lambda: ModelConstructor.get_knn_model(n_components, 2),
            'knn_v3': lambda: ModelConstructor.get_knn_model(n_components, 3),
            'nn_simple': lambda: ModelConstructor.get_nn_model(dataloader, granularity, n_components, 'simple'),
            'nn_complex': lambda: ModelConstructor.get_nn_model(dataloader, granularity, n_components, 'complex')
        }
        
        # Get the model function or raise an error if model_type is invalid
        model_func = model_map.get(model_type)
        if model_func is None:
            raise ValueError('Invalid model type')
        
        # Call the function and return the result
        return model_func()

    @staticmethod
    def get_rf_model(n_components, version):
        return RandomForestClassifier(
            n_estimators=100,
            # the below is in-line with OG code, but maybe there's a better way to set this based on num points/features than just performing optimization
            max_depth=15 if version == 1 else 5,
            min_samples_split=5 if version == 1 else 2,
            min_samples_leaf=2 if version == 1 else 1,
            random_state=ModelConstructor.RANDOM_STATE
        )

    @staticmethod
    def get_if_model(n_components, contamination):
        return IsolationForest(
            n_estimators=100,
            contamination=contamination,
            max_features=0.5,
            random_state=ModelConstructor.RANDOM_STATE
        )
    
    @staticmethod
    def get_gaussian_model(n_components, kernel):
        if kernel == 'RBF':
            kernel = 1.0 * RBF()
        elif kernel == 'matern':
            kernel = 1.0 * Matern()
        else:
            raise ValueError('Invalid kernel type')
        
        return GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=5,
            max_iter_predict=100,
            random_state=ModelConstructor.RANDOM_STATE
        )

    @staticmethod
    def get_logistic_regression_model(n_components, version):
        if version == 1:
            return LogisticRegression(
                penalty='l2',
                solver='liblinear',
                random_state=ModelConstructor.RANDOM_STATE
            )
        elif version == 2:
            return LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                l1_ratio=0.5,
                random_state=ModelConstructor.RANDOM_STATE
            )
        else:
            raise ValueError('Invalid version')
    
    @staticmethod
    def get_gbm_model(n_components):
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=ModelConstructor.RANDOM_STATE
        )
    
    @staticmethod
    def get_lightgbm_model(n_components):
        return LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            num_leaves=32,
            random_state=ModelConstructor.RANDOM_STATE
        )
    
    @staticmethod
    def get_xgboost_model(n_components):
        return XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=ModelConstructor.RANDOM_STATE
        )
    
    @staticmethod
    def get_svm_model(n_components, kernel):
        if kernel == 'rbf':
            return SVC(
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=ModelConstructor.RANDOM_STATE
            )
        elif kernel == 'poly':
            return SVC(
                kernel='poly',
                degree=3,
                gamma='scale',
                probability=True,
                random_state=ModelConstructor.RANDOM_STATE
            )
        else:
            raise ValueError('Invalid kernel type')

    @staticmethod
    def get_knn_model(n_components, version):
        if version == 1:
            return KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform'
            )
        elif version == 2:
            return KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            )
        elif version == 3:
            return KNeighborsClassifier(
                n_neighbors=15,
                weights='distance'
            )
        else:
            raise ValueError('Invalid version')

    @staticmethod
    def get_nn_model(dataloader, granularity, n_components, version):
        if n_components == -1:
            # Special case: Process different parts of the input vector with specialized blocks
            if version == 'simple':
                # Define dimensions
                one_time_info_size = len(dataloader.BLOB_ONE_TIME_INFO)
                blob_block_size = dataloader.TIMESERIES_STEPS * len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)
                num_fulldisk_blobs = dataloader.TOP_N_BLOBS
                
                # Create an input layer for the complete input
                total_input_size = one_time_info_size + num_fulldisk_blobs * blob_block_size
                inputs = tf.keras.layers.Input(shape=(total_input_size,))
                
                # Process one-time info (first section)
                one_time_info = tf.keras.layers.Lambda(lambda x: x[:, :one_time_info_size])(inputs)
                
                # First hidden layer for one-time info
                r_dimension = 4
                one_time_processed = tf.keras.layers.Dense(r_dimension, activation='relu')(one_time_info)
                one_time_processed = tf.keras.layers.BatchNormalization()(one_time_processed)
                one_time_processed = tf.keras.layers.Dropout(0.2)(one_time_processed)
                
                # Process each timeseries block with the same neural network
                # Define the shared block for processing timeseries data
                d_dimension = blob_block_size // 4
                
                # Function to create a reusable block
                def create_blob_block(input_tensor):
                    x = tf.keras.layers.Dense(d_dimension * 2, activation='relu')(input_tensor)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Dropout(0.2)(x)
                    x = tf.keras.layers.Dense(d_dimension, activation='relu')(x)
                    return x
                
                # Process each timeseries block and collect outputs
                blob_block_outputs = []
                for i in range(num_fulldisk_blobs):
                    start_idx = one_time_info_size + i * blob_block_size
                    end_idx = start_idx + blob_block_size
                    
                    # Extract this block of timeseries data
                    blob_block = tf.keras.layers.Lambda(
                        lambda x: x[:, start_idx:end_idx]
                    )(inputs)
                    
                    # Process with the same neural network block
                    processed_block = create_blob_block(blob_block)
                    blob_block_outputs.append(processed_block)
                
                # Take element-wise maximum across all processed blocks
                if len(blob_block_outputs) > 1:
                    timeseries_combined = tf.keras.layers.Maximum()(blob_block_outputs)
                else:
                    raise ValueError('Invalid number of blocks provided')
                
                # Concatenate with the one-time info results
                combined = tf.keras.layers.Concatenate()([one_time_processed, timeseries_combined])
                
                # Final dense layers
                x = tf.keras.layers.Dense(max(16, (r_dimension + d_dimension) // 2), activation='relu')(combined)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.2)(x)
                
                # Output layer
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                
                # Create model
                model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
                
            elif version == 'complex':
                # Define dimensions
                one_time_info_size = len(dataloader.BLOB_ONE_TIME_INFO)
                blob_block_size = dataloader.TIMESERIES_STEPS * len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)
                num_fulldisk_blobs = dataloader.TOP_N_BLOBS
                
                # Create an input layer for the complete input
                total_input_size = one_time_info_size + num_fulldisk_blobs * blob_block_size
                inputs = tf.keras.layers.Input(shape=(total_input_size,))
                
                # Process one-time info (first section)
                one_time_info = tf.keras.layers.Lambda(lambda x: x[:, :one_time_info_size])(inputs)
                
                # Apply normalization
                one_time_info_norm = tf.keras.layers.BatchNormalization()(one_time_info)
                
                # Determine r dimension for one-time info
                r_dimension = 4
                
                # Process one-time info with more complex structure
                x_one_time = tf.keras.layers.Dense(r_dimension, activation='relu')(one_time_info_norm)
                x_one_time = tf.keras.layers.BatchNormalization()(x_one_time)
                x_one_time = tf.keras.layers.Dropout(0.3)(x_one_time)
                
                # Second layer with residual connection
                prev_one_time = x_one_time
                x_one_time = tf.keras.layers.Dense(r_dimension, activation='relu')(x_one_time)
                x_one_time = tf.keras.layers.BatchNormalization()(x_one_time)
                x_one_time = tf.keras.layers.Dropout(0.3)(x_one_time)
                x_one_time = tf.keras.layers.Add()([x_one_time, prev_one_time])  # Residual connection
                
                # Final one-time info layer
                one_time_processed = tf.keras.layers.Dense(r_dimension, activation='relu')(x_one_time)
                one_time_processed = tf.keras.layers.BatchNormalization()(one_time_processed)
                
                # Define d dimension for timeseries data
                d_dimension = blob_block_size // 4
                
                # Function to create a reusable block with residual connections
                def create_complex_blob_block(input_tensor):
                    # Normalize input
                    x = tf.keras.layers.BatchNormalization()(input_tensor)
                    
                    # First dense layer
                    x = tf.keras.layers.Dense(d_dimension * 2, activation='relu')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Dropout(0.3)(x)
                    
                    # Second dense layer
                    prev_layer = x
                    x = tf.keras.layers.Dense(d_dimension, activation='relu')(x)
                    x = tf.keras.layers.BatchNormalization()(x)
                    x = tf.keras.layers.Dropout(0.3)(x)
                    
                    # Add residual connection with projection if needed
                    projection = tf.keras.layers.Dense(d_dimension, use_bias=False)(prev_layer)
                    x = tf.keras.layers.Add()([x, projection])
                    
                    # Final layer for this block
                    x = tf.keras.layers.Dense(d_dimension, activation='relu')(x)
                    return x
                
                # Process each timeseries block and collect outputs
                blob_block_outputs = []
                for i in range(num_fulldisk_blobs):
                    start_idx = one_time_info_size + i * blob_block_size
                    end_idx = start_idx + blob_block_size
                    
                    # Extract this block of timeseries data
                    blob_block = tf.keras.layers.Lambda(
                        lambda x: x[:, start_idx:end_idx]
                    )(inputs)
                    
                    # Process with the complex block
                    processed_block = create_complex_blob_block(blob_block)
                    blob_block_outputs.append(processed_block)
                
                # Take element-wise maximum across all processed blocks
                if len(blob_block_outputs) > 1:
                    timeseries_combined = tf.keras.layers.Maximum()(blob_block_outputs)
                else:
                    raise ValueError('Invalid number of blocks provided')
                
                # Concatenate with the one-time info results
                combined = tf.keras.layers.Concatenate()([one_time_processed, timeseries_combined])
                
                # Apply non-linear transformation to combined features
                x = tf.keras.layers.Dense(max(32, (r_dimension + d_dimension) // 2), 
                                        activation='relu')(combined)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.3)(x)
                
                # Add another layer with residual connection
                prev_combined = x
                x = tf.keras.layers.Dense(max(32, (r_dimension + d_dimension) // 4), 
                                        activation='relu')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.2)(x)
                
                # Create projection for residual if dimensions don't match
                current_dim = max(32, (r_dimension + d_dimension) // 4)
                prev_dim = max(32, (r_dimension + d_dimension) // 2)
                
                if current_dim != prev_dim:
                    projection = tf.keras.layers.Dense(current_dim, use_bias=False)(prev_combined)
                    x = tf.keras.layers.Add()([x, projection])
                else:
                    x = tf.keras.layers.Add()([x, prev_combined])
                
                # Output layer
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                
                # Create model
                model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
                
                # Add L2 regularization to all Dense layers
                for layer in model.layers:
                    if isinstance(layer, tf.keras.layers.Dense):
                        layer.kernel_regularizer = tf.keras.regularizers.l2(1e-5)
            
            else:
                raise ValueError('Invalid version')

        # n_components is not -1, so we can use the regular logic that doesn't assume any vector structure (because of PCA)
        elif granularity.startswith('per-disk'):
            # Original logic for per-disk granularity
            if version == 'simple':
                complete_input_size = n_components
                input_shape = (complete_input_size,)

                # Start with a base layer size related to n_components
                # Use a power of 2 scaling approach
                current_components = n_components
                layers = []
                
                # Add input layer
                layers.append(tf.keras.layers.Input(shape=input_shape))
                
                # Add hidden layers dynamically based on n_components
                while current_components > 4:  # Continue until we reach a small threshold
                    # Calculate neuron count for current layer (next power of 2)
                    neurons = 2 ** (current_components.bit_length())
                    
                    # Cap maximum neurons to prevent extremely large layers
                    neurons = min(neurons, 256)

                    # Add dense layer
                    layers.append(tf.keras.layers.Dense(neurons, activation='relu'))
                    layers.append(tf.keras.layers.BatchNormalization())
                    layers.append(tf.keras.layers.Dropout(0.2))

                    # Reduce component count for next layer
                    current_components = current_components // 2

                # Add output layer
                layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

                # Create model with dynamic layers
                model = Sequential(layers)

            elif version == 'complex':
                complete_input_size = n_components
                input_shape = (complete_input_size,)
                
                # Create an input layer
                inputs = tf.keras.layers.Input(shape=input_shape)
                
                # Initial normalization of inputs
                x = tf.keras.layers.BatchNormalization()(inputs)
                
                # Start with a base layer size related to n_components
                current_components = n_components
                
                # Track previous layer for a single residual connection
                previous_layer = x
                
                # Calculate the initial neuron count - capped at 128
                initial_neurons = min(256, 2 ** (current_components.bit_length() - 1))
                
                # First hidden layer
                x = tf.keras.layers.Dense(initial_neurons, activation='relu')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.3)(x)  # Higher dropout to prevent overfitting
                
                # Second hidden layer - half the neurons
                second_layer_neurons = max(16, initial_neurons // 2)
                x = tf.keras.layers.Dense(second_layer_neurons, activation='relu')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.3)(x)
                
                # Add a residual connection with dimension adaptation
                # Project previous_layer to match current dimensions if needed
                if second_layer_neurons != tf.keras.backend.int_shape(previous_layer)[-1]:
                    # Create a projection to match dimensions
                    projection = tf.keras.layers.Dense(second_layer_neurons, 
                                                    use_bias=False)(previous_layer)
                    x = tf.keras.layers.Add()([x, projection])
                else:
                    # Dimensions already match
                    x = tf.keras.layers.Add()([x, previous_layer])
                
                # Final compression layer
                x = tf.keras.layers.Dense(max(8, n_components // 4), activation='relu')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.2)(x)
                
                # Output layer
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                
                # Create model
                model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
                
                # Add mild L2 regularization to all Dense layers
                for layer in model.layers:
                    if isinstance(layer, tf.keras.layers.Dense):
                        layer.kernel_regularizer = tf.keras.regularizers.l2(1e-5)  # Lighter regularization

            else:
                raise ValueError('Invalid version')

        elif granularity == 'per-blob':
            # Original logic for per-blob granularity
            if version == 'simple':
                complete_input_size = n_components
                input_shape = (complete_input_size,)
                
                # Start with a base layer size related to n_components
                # Use a power of 2 scaling approach
                current_components = n_components
                layers = []
                
                # Add input layer
                layers.append(tf.keras.layers.Input(shape=input_shape))
                
                # Add hidden layers dynamically based on n_components
                while current_components > 4:  # Continue until we reach a small threshold
                    # Calculate neuron count for current layer (next power of 2)
                    neurons = 2 ** (current_components.bit_length())
                    
                    # Cap maximum neurons to prevent extremely large layers
                    neurons = min(neurons, 256) # TODO: Is this too high?
                    
                    # Add dense layer
                    layers.append(tf.keras.layers.Dense(neurons, activation='relu'))
                    layers.append(tf.keras.layers.BatchNormalization())
                    layers.append(tf.keras.layers.Dropout(0.2))
                    
                    # Reduce component count for next layer
                    current_components = current_components // 2
                
                # Add output layer
                layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
                
                # Create model with dynamic layers
                model = Sequential(layers)
            
            elif version == 'complex':
                complete_input_size = n_components
                input_shape = (complete_input_size,)
                
                # Create an input layer
                inputs = tf.keras.layers.Input(shape=input_shape)
                
                # Initial normalization of inputs
                x = tf.keras.layers.BatchNormalization()(inputs)
                
                # Start with a base layer size related to n_components
                current_components = n_components
                
                # Track previous layer for a single residual connection
                previous_layer = x
                
                # Calculate the initial neuron count - capped at 128
                initial_neurons = min(128, 2 ** (current_components.bit_length() - 1))
                
                # First hidden layer
                x = tf.keras.layers.Dense(initial_neurons, activation='relu')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.3)(x)  # Higher dropout to prevent overfitting
                
                # Second hidden layer - half the neurons
                second_layer_neurons = max(16, initial_neurons // 2)
                x = tf.keras.layers.Dense(second_layer_neurons, activation='relu')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.3)(x)
                
                # Add a residual connection with dimension adaptation
                # Project previous_layer to match current dimensions if needed
                if second_layer_neurons != tf.keras.backend.int_shape(previous_layer)[-1]:
                    # Create a projection to match dimensions
                    projection = tf.keras.layers.Dense(second_layer_neurons,
                                                    use_bias=False)(previous_layer)
                    x = tf.keras.layers.Add()([x, projection])
                else:
                    # Dimensions already match
                    x = tf.keras.layers.Add()([x, previous_layer])
                
                # Final compression layer
                x = tf.keras.layers.Dense(max(8, n_components // 4), activation='relu')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.2)(x)
                
                # Output layer
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                
                # Create model
                model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
                
                # Add mild L2 regularization to all Dense layers
                for layer in model.layers:
                    if isinstance(layer, tf.keras.layers.Dense):
                        layer.kernel_regularizer = tf.keras.regularizers.l2(1e-5)  # Lighter regularization
                
            else:
                raise ValueError('Invalid version')

        # Compile model with the same metrics for all cases
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        model.summary()
        return model
