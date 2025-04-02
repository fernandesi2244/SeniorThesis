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
import VolumeSlicesAndCubeDataLoader
import VolumeCubeDataLoader
import VolumeSlicesDataLoader

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

        Non-convolutional models:
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

        For volume convolutions:
        'conv_nn_on_slices_and_cube',
        'conv_nn_on_cube',
        'conv_nn_on_cube_simple',
        'conv_nn_on_cube_complex',
        'conv_nn_on_slices',
        """

        if dataloader_type == 'photospheric':
            dataloader = PhotosphericDataLoader.SEPInputDataGenerator
        elif dataloader_type == 'coronal':
            dataloader = CoronalDataLoader.SEPInputDataGenerator
        elif dataloader_type == 'numeric':
            dataloader = NumericDataLoader.SEPInputDataGenerator
        elif dataloader_type == 'slices_and_cube':
            dataloader = VolumeSlicesAndCubeDataLoader.SecondarySEPInputDataGenerator
        elif dataloader_type == 'cube':
            dataloader = VolumeCubeDataLoader.SecondarySEPInputDataGenerator
        elif dataloader_type == 'slices':
            dataloader = VolumeSlicesDataLoader.SecondarySEPInputDataGenerator
        else:
            raise ValueError('Invalid dataloader type')

        # Dictionary mapping model types to their corresponding functions
        model_map = {
            # Non-convolutional models
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
            'nn_complex': lambda: ModelConstructor.get_nn_model(dataloader, granularity, n_components, 'complex'),

            # Convolutional models for volume data
            'conv_nn_on_slices_and_cube': lambda: ModelConstructor.get_conv_nn(dataloader, granularity, 'slices_and_cube'),
            'conv_nn_on_cube': lambda: ModelConstructor.get_conv_nn(dataloader, granularity, 'cube'),
            'conv_nn_on_cube_simple': lambda: ModelConstructor.get_conv_nn(dataloader, granularity, 'cube_simple'),
            'conv_nn_on_cube_complex': lambda: ModelConstructor.get_conv_nn(dataloader, granularity, 'cube_complex'),
            'conv_nn_on_slices': lambda: ModelConstructor.get_conv_nn(dataloader, granularity, 'slices'),
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
            # Make assumption we're dealing with per-disk granularity
            assert granularity.startswith('per-disk')
            
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
    
    def get_conv_nn(dataloader, granularity, version):
        if granularity == 'per-blob':
            raise ValueError('Convolutional models are not yet supported for per-blob granularity')
        
        # NOTE: From now on, assuming per-disk. Moreover, both options (1d vs 4hr) are treated equally
        class StackLayer(tf.keras.layers.Layer):
            def __init__(self, axis=1, **kwargs):
                super(StackLayer, self).__init__(**kwargs)
                self.axis = axis
                
            def call(self, inputs):
                return tf.stack(inputs, axis=self.axis)
        
        if version == 'slices_and_cube':
            """
            Input format per batch example (all as flattened scalar values immediately following each other)
            - BLOB_ONE_TIME_INFO
            x TOP_N_BLOBS
                -- BLOB_VECTOR_COLUMNS_GENERAL
            X TOP_N_BLOBS
                -- 5*nx*ny*channels (5 xy slices)
                -- 5*nx*nz*channels (5 xz slices)
                -- 5*ny*nz*channels (5 yz slices)
                -- 5*5*5*channels (5x5x5 cube)
            
            where the first-order elements of the two TOP_N_BLOB lists correspond to the same blob.
            """

            nx, ny, nz, channels = 200, 400, 100, 3

            complete_input_size = len(dataloader.BLOB_ONE_TIME_INFO) + dataloader.TOP_N_BLOBS * (
                len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + 5 * nx * ny * channels + 5 * nx * nz * channels +
                5 * ny * nz * channels + 5 * 5 * 5 * channels)
            
            print('Complete input size:', complete_input_size)
            
            flattened_input = tf.keras.layers.Input(shape=(complete_input_size,))

            # Split the input into the two main parts
            one_time_info_input = tf.keras.layers.Lambda(lambda x: x[:, :len(dataloader.BLOB_ONE_TIME_INFO)])(flattened_input)
            blob_data_input = tf.keras.layers.Lambda(lambda x: x[:, len(dataloader.BLOB_ONE_TIME_INFO):])(flattened_input)

            print('One time info input shape:', tf.keras.backend.int_shape(one_time_info_input))
            print('Blob data input shape:', tf.keras.backend.int_shape(blob_data_input))

            # Process the one-time info, bringing it down to 2 neurons
            one_time_info_output = tf.keras.layers.Dense(2, activation='relu')(one_time_info_input)
            one_time_info_output = tf.keras.layers.BatchNormalization()(one_time_info_output)

            print('Got to transforming blob input')

            # Transform the blob_data_input to the desired format
            new_blob_data_input = []
            for i in range(dataloader.TOP_N_BLOBS):
                start_general_index = i * len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)
                end_general_index = start_general_index + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)
                general_info = tf.keras.layers.Lambda(lambda x: x[:, start_general_index:end_general_index])(blob_data_input)

                start_slices_index = dataloader.TOP_N_BLOBS * len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + i * (
                        5 * nx * ny * channels + 5 * nx * nz * channels + 5 * ny * nz * channels + 5 * 5 * 5 * channels)
                end_cube_index = start_slices_index + 5 * nx * ny * channels + 5 * nx * nz * channels + 5 * ny * nz * channels + 5 * 5 * 5 * channels
                slices_and_cube = tf.keras.layers.Lambda(lambda x: x[:, start_slices_index:end_cube_index])(blob_data_input)

                new_blob_data_input.append(tf.keras.layers.Concatenate()([general_info, slices_and_cube]))
            
            # Print the shape of the new blob data input
            print("Shape of new blob data input:", [tf.keras.backend.int_shape(blob) for blob in new_blob_data_input])
            
            print('Got to shared layer creation')

            # Create shared layers for processing blobs
            # Shared layers for blob general info
            general_dense = tf.keras.layers.Dense(2, activation='relu')
            general_bn = tf.keras.layers.BatchNormalization()
            
            # Shared layers for xy slices
            xy_conv_layer = tf.keras.layers.Conv2D(4, (5, 5), strides=(2, 2), activation='relu')
            xy_bn_layer = tf.keras.layers.BatchNormalization()
            xy_pooling_layer = tf.keras.layers.MaxPooling2D((4, 4))
            xy_dense = tf.keras.layers.Dense(8, activation='relu')
            xy_bn = tf.keras.layers.BatchNormalization()
            
            # Shared layers for xz slices
            xz_conv_layer = tf.keras.layers.Conv2D(4, (5, 5), strides=(2, 2), activation='relu')
            xz_bn_layer = tf.keras.layers.BatchNormalization()
            xz_pooling_layer = tf.keras.layers.MaxPooling2D((4, 4))
            xz_dense = tf.keras.layers.Dense(8, activation='relu')
            xz_bn = tf.keras.layers.BatchNormalization()
            
            # Shared layers for yz slices
            yz_conv_layer = tf.keras.layers.Conv2D(4, (5, 5), strides=(2, 2), activation='relu')
            yz_bn_layer = tf.keras.layers.BatchNormalization()
            yz_pooling_layer = tf.keras.layers.MaxPooling2D((4, 4))
            yz_dense = tf.keras.layers.Dense(8, activation='relu')
            yz_bn = tf.keras.layers.BatchNormalization()
            
            # Shared layers for cube
            cube_conv_layer = tf.keras.layers.Conv3D(4, (3, 3, 3), activation='relu')
            cube_bn_layer = tf.keras.layers.BatchNormalization()
            cube_dense = tf.keras.layers.Dense(8, activation='relu')
            cube_bn = tf.keras.layers.BatchNormalization()
            
            # Shared layers for final blob processing
            combined_dense = tf.keras.layers.Dense(8, activation='relu')
            combined_bn = tf.keras.layers.BatchNormalization()
            
            # Process each blob with shared layers
            blob_data_output = []
            len_each_blob = len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + 5 * nx * ny * channels + 5 * nx * nz * channels + 5 * ny * nz * channels + 5 * 5 * 5 * channels
            
            print('Got to for loop for blobs across day')

            for i in range(dataloader.TOP_N_BLOBS):
                blob = new_blob_data_input[i]
                
                # Extract the blob data for this blob
                start_idx = 0  # index w.r.t. individual blob tensor
                
                # Extract and process general info
                curr_blob_general_info = tf.keras.layers.Lambda(lambda x: x[:, start_idx:start_idx + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)])(blob)
                print(f'Start to end index for blob {i} general info:', start_idx, start_idx + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL))
                curr_blob_general_info_output = general_dense(curr_blob_general_info)
                curr_blob_general_info_output = general_bn(curr_blob_general_info_output)
                
                print('Got to process xy slices')

                # Process xy slices
                xy_outputs = []
                for curr_slice in range(5):
                    slice_j_start = start_idx + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + curr_slice * nx * ny * channels
                    slice_j_end = slice_j_start + nx * ny * channels
                    print(f'Start to end index for blob {i} xy slice {curr_slice}:', slice_j_start, slice_j_end)
                    xy_slice = tf.keras.layers.Lambda(lambda x: x[:, slice_j_start:slice_j_end])(blob)
                    
                    # Reshape for Conv2D
                    xy_slice = tf.keras.layers.Reshape((nx, ny, channels))(xy_slice)
                    
                    # Apply shared convolution
                    xy_slice_output = xy_conv_layer(xy_slice)
                    xy_slice_output = xy_bn_layer(xy_slice_output)
                    xy_slice_output = xy_pooling_layer(xy_slice_output)
                    xy_slice_output = tf.keras.layers.Flatten()(xy_slice_output)
                    
                    xy_outputs.append(xy_slice_output)
                
                # Take elementwise max across all xy slice outputs
                xy_output = tf.keras.layers.Maximum()(xy_outputs)
                xy_output = xy_dense(xy_output)
                xy_output = xy_bn(xy_output)

                print('Got to process xz slices')
                
                # Process xz slices
                xz_outputs = []
                for curr_slice in range(5):
                    slice_j_start = start_idx + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + 5 * nx * ny * channels + curr_slice * nx * nz * channels
                    slice_j_end = slice_j_start + nx * nz * channels
                    print(f'Start to end index for blob {i} xz slice {curr_slice}:', slice_j_start, slice_j_end)
                    xz_slice = tf.keras.layers.Lambda(lambda x: x[:, slice_j_start:slice_j_end])(blob)
                    
                    # Reshape for Conv2D
                    xz_slice = tf.keras.layers.Reshape((nx, nz, channels))(xz_slice)
                    
                    # Apply shared convolution
                    xz_slice_output = xz_conv_layer(xz_slice)
                    xz_slice_output = xz_bn_layer(xz_slice_output)
                    xz_slice_output = xz_pooling_layer(xz_slice_output)
                    xz_slice_output = tf.keras.layers.Flatten()(xz_slice_output)
                    
                    xz_outputs.append(xz_slice_output)
                
                # Take elementwise max across all xz slice outputs
                xz_output = tf.keras.layers.Maximum()(xz_outputs)
                xz_output = xz_dense(xz_output)
                xz_output = xz_bn(xz_output)

                print('Got to process yz slices')
                
                # Process yz slices
                yz_outputs = []
                for curr_slice in range(5):
                    slice_j_start = start_idx + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + 5 * nx * ny * channels + 5 * nx * nz * channels + curr_slice * ny * nz * channels
                    slice_j_end = slice_j_start + ny * nz * channels
                    print(f'Start to end index for blob {i} yz slice {curr_slice}:', slice_j_start, slice_j_end)
                    yz_slice = tf.keras.layers.Lambda(lambda x: x[:, slice_j_start:slice_j_end])(blob)
                    
                    # Reshape for Conv2D
                    yz_slice = tf.keras.layers.Reshape((ny, nz, channels))(yz_slice)
                    
                    # Apply shared convolution
                    yz_slice_output = yz_conv_layer(yz_slice)
                    yz_slice_output = yz_bn_layer(yz_slice_output)
                    yz_slice_output = yz_pooling_layer(yz_slice_output)
                    yz_slice_output = tf.keras.layers.Flatten()(yz_slice_output)
                    
                    yz_outputs.append(yz_slice_output)
                
                # Take elementwise max across all yz slice outputs
                yz_output = tf.keras.layers.Maximum()(yz_outputs)
                yz_output = yz_dense(yz_output)
                yz_output = yz_bn(yz_output)

                print('Got to process cube part')
                
                # Process cube
                cube_start = start_idx + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + 5 * nx * ny * channels + 5 * nx * nz * channels + 5 * ny * nz * channels
                cube_end = cube_start + 5 * 5 * 5 * channels
                print(f'Start to end index for blob {i} cube:', cube_start, cube_end)
                cube = tf.keras.layers.Lambda(lambda x: x[:, cube_start:cube_end])(blob)
                print("Shape of cube:", tf.keras.backend.int_shape(cube))
                cube = tf.keras.layers.Reshape((5, 5, 5, channels))(cube)
                print("Shape of cube after reshape:", tf.keras.backend.int_shape(cube))
                
                cube_output = cube_conv_layer(cube)
                cube_output = cube_bn_layer(cube_output)
                cube_output = tf.keras.layers.Flatten()(cube_output)
                cube_output = cube_dense(cube_output)
                cube_output = cube_bn(cube_output)
                
                # Concatenate all the outputs for this blob
                combined_output = tf.keras.layers.Concatenate()([curr_blob_general_info_output, xy_output, xz_output, yz_output, cube_output])
                
                # Process the combined output
                combined_output = combined_dense(combined_output)
                combined_output = combined_bn(combined_output)
                
                # Add the output to the list
                blob_data_output.append(combined_output)
            
            # Aggregate the outputs for each blob by taking the maximum value for each neuron across all blobs
            blob_data_output = tf.keras.layers.Maximum()(blob_data_output)
            
            # Combine the one-time info and blob data outputs
            combined_output = tf.keras.layers.Concatenate()([one_time_info_output, blob_data_output])
            
            # Final processing layers
            combined_output = tf.keras.layers.Dense(4, activation='relu')(combined_output)
            combined_output = tf.keras.layers.BatchNormalization()(combined_output)
            
            # Output layer
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(combined_output)
            
            # Create model
            model = tf.keras.models.Model(inputs=flattened_input, outputs=outputs)
            
            # Compile model with the same metrics for all cases
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            model.summary()
            
            return model
        if version == 'cube':
            """
            Input format per batch example (all as flattened scalar values immediately following each other)
            - BLOB_ONE_TIME_INFO
            x TOP_N_BLOBS
                -- BLOB_VECTOR_COLUMNS_GENERAL
                -- 5*5*5*channels (5x5x5 cube)
            """
            nx, ny, nz, channels = 200, 400, 100, 3

            complete_input_size = len(dataloader.BLOB_ONE_TIME_INFO) + dataloader.TOP_N_BLOBS * (
                len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + 5 * 5 * 5 * channels)
            
            flattened_input = tf.keras.layers.Input(shape=(complete_input_size,))

            # Split the input into the two main parts
            one_time_info_input = tf.keras.layers.Lambda(lambda x: x[:, :len(dataloader.BLOB_ONE_TIME_INFO)])(flattened_input)
            blob_data_input = tf.keras.layers.Lambda(lambda x: x[:, len(dataloader.BLOB_ONE_TIME_INFO):])(flattened_input)

            # Process the one-time info, bringing it down to 2 neurons
            one_time_info_output = tf.keras.layers.Dense(2, activation='relu')(one_time_info_input)
            one_time_info_output = tf.keras.layers.BatchNormalization()(one_time_info_output)

            # Transform the blob_data_input to the desired format
            new_blob_data_input = []
            for i in range(dataloader.TOP_N_BLOBS):
                start_general_index = i * len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)
                end_general_index = start_general_index + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)
                general_info = tf.keras.layers.Lambda(lambda x: x[:, start_general_index:end_general_index])(blob_data_input)

                start_cube_index = dataloader.TOP_N_BLOBS * len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + i * (5 * 5 * 5 * channels)
                end_cube_index = start_cube_index + 5 * 5 * 5 * channels
                cube = tf.keras.layers.Lambda(lambda x: x[:, start_cube_index:end_cube_index])(blob_data_input)

                new_blob_data_input.append(tf.keras.layers.Concatenate()([general_info, cube]))
            
            # Create shared layers for all blobs
            # Shared layers for blob general info
            general_dense = tf.keras.layers.Dense(2, activation='relu')
            general_bn = tf.keras.layers.BatchNormalization()
            
            # Shared layers for cube
            cube_conv_layer = tf.keras.layers.Conv3D(4, (3, 3, 3), activation='relu')
            cube_bn_layer = tf.keras.layers.BatchNormalization()
            cube_dense = tf.keras.layers.Dense(8, activation='relu')
            cube_bn = tf.keras.layers.BatchNormalization()
            
            # Shared layers for final blob processing
            combined_dense = tf.keras.layers.Dense(8, activation='relu')
            combined_bn = tf.keras.layers.BatchNormalization()
            
            # Process each blob with shared layers
            blob_data_output = []
            len_each_blob = len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + 5 * 5 * 5 * channels
            
            for i in range(dataloader.TOP_N_BLOBS):
                blob = new_blob_data_input[i]
                
                # Extract the blob data for this blob
                start_idx = 0  # index w.r.t. individual blob tensor
                
                # Extract and process general info
                curr_blob_general_info = tf.keras.layers.Lambda(lambda x: x[:, start_idx:start_idx + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)])(blob)
                curr_blob_general_info_output = general_dense(curr_blob_general_info)
                curr_blob_general_info_output = general_bn(curr_blob_general_info_output)
                
                # Process cube
                cube_start = start_idx + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)
                cube_end = cube_start + 5 * 5 * 5 * channels
                cube = tf.keras.layers.Lambda(lambda x: x[:, cube_start:cube_end])(blob)
                cube = tf.keras.layers.Reshape((5, 5, 5, channels))(cube)
                
                cube_output = cube_conv_layer(cube)
                cube_output = cube_bn_layer(cube_output)
                cube_output = tf.keras.layers.Flatten()(cube_output)
                cube_output = cube_dense(cube_output)
                cube_output = cube_bn(cube_output)
                
                # Concatenate all the outputs for this blob
                combined_output = tf.keras.layers.Concatenate()([curr_blob_general_info_output, cube_output])
                
                # Process the combined output
                combined_output = combined_dense(combined_output)
                combined_output = combined_bn(combined_output)
                
                # Add the output to the list
                blob_data_output.append(combined_output)
            
            # Aggregate the outputs for each blob by taking the maximum value for each neuron across all blobs
            blob_data_output = tf.keras.layers.Maximum()(blob_data_output)
            
            # Combine the one-time info and blob data outputs
            combined_output = tf.keras.layers.Concatenate()([one_time_info_output, blob_data_output])
            
            # Final processing layers
            combined_output = tf.keras.layers.Dense(4, activation='relu')(combined_output)
            combined_output = tf.keras.layers.BatchNormalization()(combined_output)
            
            # Output layer
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(combined_output)
            
            # Create model
            model = tf.keras.models.Model(inputs=flattened_input, outputs=outputs)
            
            # Compile model with the same metrics for all cases
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            model.summary()
            
            return model
        elif version == 'cube_simple':
            """
            Input format per batch example (all as flattened scalar values immediately following each other)
            - BLOB_ONE_TIME_INFO
            x TOP_N_BLOBS
                -- BLOB_VECTOR_COLUMNS_GENERAL
                -- 5*5*5*channels (5x5x5 cube)
            """
            nx, ny, nz, channels = 200, 400, 100, 3

            complete_input_size = len(dataloader.BLOB_ONE_TIME_INFO) + dataloader.TOP_N_BLOBS * (
                len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + 5 * 5 * 5 * channels)
            
            flattened_input = tf.keras.layers.Input(shape=(complete_input_size,))

            # Split the input into the two main parts
            one_time_info_input = tf.keras.layers.Lambda(lambda x: x[:, :len(dataloader.BLOB_ONE_TIME_INFO)])(flattened_input)
            blob_data_input = tf.keras.layers.Lambda(lambda x: x[:, len(dataloader.BLOB_ONE_TIME_INFO):])(flattened_input)

            # Process the one-time info (single small layer, no BatchNorm)
            one_time_info_output = tf.keras.layers.Dense(1, activation='relu')(one_time_info_input)
            
            # Create shared layers for all blobs (much smaller)
            # Shared layer for blob general info
            general_dense = tf.keras.layers.Dense(1, activation='relu')
            
            # Simplified shared layers for cube (fewer filters, no BatchNorm)
            cube_conv_layer = tf.keras.layers.Conv3D(2, (3, 3, 3), activation='relu')
            cube_dense = tf.keras.layers.Dense(2, activation='relu')
            
            # Process each blob with shared layers
            blob_data_output = []
            len_each_blob = len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + 5 * 5 * 5 * channels
            
            for i in range(dataloader.TOP_N_BLOBS):
                # Extract the blob data for this blob
                start_general_index = i * len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)
                end_general_index = start_general_index + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)
                general_info = tf.keras.layers.Lambda(lambda x: x[:, start_general_index:end_general_index])(blob_data_input)
                
                # Extract cube data
                start_cube_index = dataloader.TOP_N_BLOBS * len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + i * (5 * 5 * 5 * channels)
                end_cube_index = start_cube_index + 5 * 5 * 5 * channels
                cube = tf.keras.layers.Lambda(lambda x: x[:, start_cube_index:end_cube_index])(blob_data_input)
                
                # Process general info (single small layer)
                general_output = general_dense(general_info)
                
                # Process cube (single conv layer, single small dense layer)
                cube = tf.keras.layers.Reshape((5, 5, 5, channels))(cube)
                cube_output = cube_conv_layer(cube)
                cube_output = tf.keras.layers.Flatten()(cube_output)
                cube_output = cube_dense(cube_output)
                
                # Concatenate outputs for this blob
                combined_output = tf.keras.layers.Concatenate()([general_output, cube_output])
                
                # Add the output to the list
                blob_data_output.append(combined_output)
            
            # Aggregate the outputs for each blob by taking the maximum value
            blob_data_output = tf.keras.layers.Maximum()(blob_data_output)
            
            # Combine the one-time info and blob data outputs
            combined_output = tf.keras.layers.Concatenate()([one_time_info_output, blob_data_output])
            
            # Final processing (single small layer)
            combined_output = tf.keras.layers.Dense(2, activation='relu')(combined_output)
            
            # Output layer
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(combined_output)
            
            # Create model
            model = tf.keras.models.Model(inputs=flattened_input, outputs=outputs)
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
            
            model.summary()
            
            return model
        elif version == 'cube_complex':
            """
            Input format per batch example (all as flattened scalar values immediately following each other)
            - BLOB_ONE_TIME_INFO
            x TOP_N_BLOBS
                -- BLOB_VECTOR_COLUMNS_GENERAL
                -- 5*5*5*channels (5x5x5 cube)
            """
            nx, ny, nz, channels = 200, 400, 100, 3

            complete_input_size = len(dataloader.BLOB_ONE_TIME_INFO) + dataloader.TOP_N_BLOBS * (
                len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + 5 * 5 * 5 * channels)
            
            flattened_input = tf.keras.layers.Input(shape=(complete_input_size,))

            # Split the input into the two main parts
            one_time_info_input = tf.keras.layers.Lambda(lambda x: x[:, :len(dataloader.BLOB_ONE_TIME_INFO)])(flattened_input)
            blob_data_input = tf.keras.layers.Lambda(lambda x: x[:, len(dataloader.BLOB_ONE_TIME_INFO):])(flattened_input)

            # Process the one-time info with a small network
            one_time_info_output = tf.keras.layers.Dense(8, activation='relu')(one_time_info_input)
            one_time_info_output = tf.keras.layers.BatchNormalization()(one_time_info_output)

            # Create moderate-sized shared layers for all blobs
            # Shared layers for blob general info
            general_dense = tf.keras.layers.Dense(4, activation='relu')
            general_bn = tf.keras.layers.BatchNormalization()
            
            # Shared layers for cube - small conv layer
            # Using only 8 filters to keep parameter count down
            cube_conv_layer = tf.keras.layers.Conv3D(8, (3, 3, 3), padding='same', activation='relu')
            cube_bn = tf.keras.layers.BatchNormalization()
            cube_dense = tf.keras.layers.Dense(12, activation='relu')
            
            # Process each blob with shared layers
            blob_data_output = []
            
            for i in range(dataloader.TOP_N_BLOBS):
                # Extract the blob data for this blob
                start_general_index = i * len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)
                end_general_index = start_general_index + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)
                general_info = tf.keras.layers.Lambda(lambda x: x[:, start_general_index:end_general_index])(blob_data_input)
                
                # Extract cube data
                start_cube_index = dataloader.TOP_N_BLOBS * len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + i * (5 * 5 * 5 * channels)
                end_cube_index = start_cube_index + 5 * 5 * 5 * channels
                cube = tf.keras.layers.Lambda(lambda x: x[:, start_cube_index:end_cube_index])(blob_data_input)
                
                # Process general info
                general_out = general_dense(general_info)
                general_out = general_bn(general_out)
                
                # Process cube
                cube = tf.keras.layers.Reshape((5, 5, 5, channels))(cube)
                cube_out = cube_conv_layer(cube)
                cube_out = cube_bn(cube_out)
                
                # Global pooling to drastically reduce parameters
                cube_out = tf.keras.layers.GlobalAveragePooling3D()(cube_out)
                cube_out = cube_dense(cube_out)
                
                # Concatenate the outputs for this blob
                combined_output = tf.keras.layers.Concatenate()([general_out, cube_out])
                
                # Add to blob outputs list
                blob_data_output.append(combined_output)
            
            # Simple max pooling across blobs
            blob_features = tf.keras.layers.Maximum()(blob_data_output)
            
            # Combine the one-time info and blob data outputs
            combined_output = tf.keras.layers.Concatenate()([one_time_info_output, blob_features])
            
            # Final processing layer - keeping it small
            combined_output = tf.keras.layers.Dense(10, activation='relu')(combined_output)
            combined_output = tf.keras.layers.BatchNormalization()(combined_output)
            
            # Output layer
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(combined_output)
            
            # Create model
            model = tf.keras.models.Model(inputs=flattened_input, outputs=outputs)
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
            
            # Print model summary to verify parameter count
            model.summary()
            
            return model
        elif version == 'slices':
            """
            Input format per batch example (all as flattened scalar values immediately following each other)
            - BLOB_ONE_TIME_INFO
            x TOP_N_BLOBS
                -- BLOB_VECTOR_COLUMNS_GENERAL
                -- 5*nx*ny*channels (5 xy slices)
                -- 5*nx*nz*channels (5 xz slices)
                -- 5*ny*nz*channels (5 yz slices)
            """
            nx, ny, nz, channels = 200, 400, 100, 3

            complete_input_size = len(dataloader.BLOB_ONE_TIME_INFO) + dataloader.TOP_N_BLOBS * (
                len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + 5 * nx * ny * channels + 5 * nx * nz * channels +
                5 * ny * nz * channels)
            
            flattened_input = tf.keras.layers.Input(shape=(complete_input_size,))

            # Split the input into the two main parts
            one_time_info_input = tf.keras.layers.Lambda(lambda x: x[:, :len(dataloader.BLOB_ONE_TIME_INFO)])(flattened_input)
            blob_data_input = tf.keras.layers.Lambda(lambda x: x[:, len(dataloader.BLOB_ONE_TIME_INFO):])(flattened_input)

            # Process the one-time info, bringing it down to 2 neurons
            one_time_info_output = tf.keras.layers.Dense(2, activation='relu')(one_time_info_input)
            one_time_info_output = tf.keras.layers.BatchNormalization()(one_time_info_output)

            # Transform the blob_data_input to the desired format
            new_blob_data_input = []
            for i in range(dataloader.TOP_N_BLOBS):
                start_general_index = i * len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)
                end_general_index = start_general_index + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)
                general_info = tf.keras.layers.Lambda(lambda x: x[:, start_general_index:end_general_index])(blob_data_input)

                start_slices_index = dataloader.TOP_N_BLOBS * len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + i * (
                        5 * nx * ny * channels + 5 * nx * nz * channels + 5 * ny * nz * channels)
                end_slices_index = start_slices_index + 5 * nx * ny * channels + 5 * nx * nz * channels + 5 * ny * nz * channels
                slices = tf.keras.layers.Lambda(lambda x: x[:, start_slices_index:end_slices_index])(blob_data_input)

                new_blob_data_input.append(tf.keras.layers.Concatenate()([general_info, slices]))
            
            # Create shared layers for processing blobs
            # Shared layers for blob general info
            general_dense = tf.keras.layers.Dense(2, activation='relu')
            general_bn = tf.keras.layers.BatchNormalization()
            
            # Shared layers for xy slices
            xy_conv_layer = tf.keras.layers.Conv2D(4, (5, 5), strides=(2, 2), activation='relu')
            xy_bn_layer = tf.keras.layers.BatchNormalization()
            xy_pooling_layer = tf.keras.layers.MaxPooling2D((4, 4))
            xy_dense = tf.keras.layers.Dense(8, activation='relu')
            xy_bn = tf.keras.layers.BatchNormalization()
            
            # Shared layers for xz slices
            xz_conv_layer = tf.keras.layers.Conv2D(4, (5, 5), strides=(2, 2), activation='relu')
            xz_bn_layer = tf.keras.layers.BatchNormalization()
            xz_pooling_layer = tf.keras.layers.MaxPooling2D((4, 4))
            xz_dense = tf.keras.layers.Dense(8, activation='relu')
            xz_bn = tf.keras.layers.BatchNormalization()
            
            # Shared layers for yz slices
            yz_conv_layer = tf.keras.layers.Conv2D(4, (5, 5), strides=(2, 2), activation='relu')
            yz_bn_layer = tf.keras.layers.BatchNormalization()
            yz_pooling_layer = tf.keras.layers.MaxPooling2D((4, 4))
            yz_dense = tf.keras.layers.Dense(8, activation='relu')
            yz_bn = tf.keras.layers.BatchNormalization()
            
            # Shared layers for final blob processing
            combined_dense = tf.keras.layers.Dense(8, activation='relu')
            combined_bn = tf.keras.layers.BatchNormalization()
            
            # Process each blob with shared layers
            blob_data_output = []
            len_each_blob = len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + 5 * nx * ny * channels + 5 * nx * nz * channels + 5 * ny * nz * channels
            
            for i in range(dataloader.TOP_N_BLOBS):
                blob = new_blob_data_input[i]
                
                # Extract the blob data for this blob
                start_idx = 0  # index w.r.t. individual blob tensor
                
                # Extract and process general info
                curr_blob_general_info = tf.keras.layers.Lambda(lambda x: x[:, start_idx:start_idx + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL)])(blob)
                curr_blob_general_info_output = general_dense(curr_blob_general_info)
                curr_blob_general_info_output = general_bn(curr_blob_general_info_output)
                
                # Process xy slices
                xy_outputs = []
                for curr_slice in range(5):
                    slice_j_start = start_idx + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + curr_slice * nx * ny * channels
                    slice_j_end = slice_j_start + nx * ny * channels
                    xy_slice = tf.keras.layers.Lambda(lambda x: x[:, slice_j_start:slice_j_end])(blob)
                    
                    # Reshape for Conv2D
                    xy_slice = tf.keras.layers.Reshape((nx, ny, channels))(xy_slice)
                    
                    # Apply shared convolution
                    xy_slice_output = xy_conv_layer(xy_slice)
                    xy_slice_output = xy_bn_layer(xy_slice_output)
                    xy_slice_output = xy_pooling_layer(xy_slice_output)
                    xy_slice_output = tf.keras.layers.Flatten()(xy_slice_output)
                    
                    xy_outputs.append(xy_slice_output)
                
                # Take elementwise max across all xy slice outputs
                xy_output = tf.keras.layers.Maximum()(xy_outputs)
                xy_output = xy_dense(xy_output)
                xy_output = xy_bn(xy_output)
                
                # Process xz slices
                xz_outputs = []
                for curr_slice in range(5):
                    slice_j_start = start_idx + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + 5 * nx * ny * channels + curr_slice * nx * nz * channels
                    slice_j_end = slice_j_start + nx * nz * channels
                    xz_slice = tf.keras.layers.Lambda(lambda x: x[:, slice_j_start:slice_j_end])(blob)
                    
                    # Reshape for Conv2D
                    xz_slice = tf.keras.layers.Reshape((nx, nz, channels))(xz_slice)
                    
                    # Apply shared convolution
                    xz_slice_output = xz_conv_layer(xz_slice)
                    xz_slice_output = xz_bn_layer(xz_slice_output)
                    xz_slice_output = xz_pooling_layer(xz_slice_output)
                    xz_slice_output = tf.keras.layers.Flatten()(xz_slice_output)
                    
                    xz_outputs.append(xz_slice_output)
                
                # Take elementwise max across all xz slice outputs
                xz_output = tf.keras.layers.Maximum()(xz_outputs)
                xz_output = xz_dense(xz_output)
                xz_output = xz_bn(xz_output)
                
                # Process yz slices
                yz_outputs = []
                for curr_slice in range(5):
                    slice_j_start = start_idx + len(dataloader.BLOB_VECTOR_COLUMNS_GENERAL) + 5 * nx * ny * channels + 5 * nx * nz * channels + curr_slice * ny * nz * channels
                    slice_j_end = slice_j_start + ny * nz * channels
                    yz_slice = tf.keras.layers.Lambda(lambda x: x[:, slice_j_start:slice_j_end])(blob)
                    
                    # Reshape for Conv2D
                    yz_slice = tf.keras.layers.Reshape((ny, nz, channels))(yz_slice)
                    
                    # Apply shared convolution
                    yz_slice_output = yz_conv_layer(yz_slice)
                    yz_slice_output = yz_bn_layer(yz_slice_output)
                    yz_slice_output = yz_pooling_layer(yz_slice_output)
                    yz_slice_output = tf.keras.layers.Flatten()(yz_slice_output)
                    
                    yz_outputs.append(yz_slice_output)
                
                # Take elementwise max across all yz slice outputs
                yz_output = tf.keras.layers.Maximum()(yz_outputs)
                yz_output = yz_dense(yz_output)
                yz_output = yz_bn(yz_output)
                
                # Concatenate all the outputs for this blob
                combined_output = tf.keras.layers.Concatenate()([curr_blob_general_info_output, xy_output, xz_output, yz_output])
                
                # Process the combined output
                combined_output = combined_dense(combined_output)
                combined_output = combined_bn(combined_output)
                
                # Add the output to the list
                blob_data_output.append(combined_output)
            
            # Aggregate the outputs for each blob by taking the maximum value for each neuron across all blobs
            blob_data_output = tf.keras.layers.Maximum()(blob_data_output)
            
            # Combine the one-time info and blob data outputs
            combined_output = tf.keras.layers.Concatenate()([one_time_info_output, blob_data_output])
            
            # Final processing layers
            combined_output = tf.keras.layers.Dense(4, activation='relu')(combined_output)
            combined_output = tf.keras.layers.BatchNormalization()(combined_output)
            
            # Output layer
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(combined_output)
            
            # Create model
            model = tf.keras.models.Model(inputs=flattened_input, outputs=outputs)
            
            # Compile model with the same metrics for all cases
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            model.summary()
            
            return model
        else:
            raise ValueError('Invalid version')
