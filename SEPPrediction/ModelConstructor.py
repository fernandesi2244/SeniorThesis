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

class ModelConstructor(object):
    RANDOM_STATE = 42

    # TODO: Add another mode where you can disable PCA. This will
    # let us test more complex models that are even more interpretable
    # in some sense because we can perform convolution just on certain
    # parts of the input, for example.
    @classmethod
    def create_model(model_type, granularity, n_components):
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
        match model_type:
            case 'random_forest_simple':
                return get_rf_model(n_components, 1)
            case 'random_forest_complex':
                return get_rf_model(n_components, 2)
            case 'isolation_forest':
                return get_if_model(n_components)
            case 'gaussian_RBF':
                return get_gaussian_model(n_components, 'RBF')
            case 'gaussian_matern':
                return get_gaussian_model(n_components, 'matern')
            case 'logistic_regression_v1':
                return get_logistic_regression_model(n_components, 1)
            case 'logistic_regression_v2':
                return get_logistic_regression_model(n_components, 2)
            case 'gbm':
                return get_gbm_model(n_components)
            case 'lightgbm':
                return get_lightgbm_model(n_components)
            case 'xgboost':
                return get_xgboost_model(n_components)
            case 'svm_rbf':
                return get_svm_model(n_components, 'rbf')
            case 'svm_poly':
                return get_svm_model(n_components, 'poly')
            case 'knn_v1':
                return get_knn_model(n_components, 1)
            case 'knn_v2':
                return get_knn_model(n_components, 2)
            case 'knn_v3':
                return get_knn_model(n_components, 3)
            case 'nn_simple':
                return get_nn_model(granularity, n_components, 'simple')
            case 'nn_complex':
                return get_nn_model(granularity, n_components, 'complex')
            case _:
                raise ValueError('Invalid model type')

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
    def get_if_model(n_components):
        return IsolationForest(
            n_estimators=100,
            contamination='auto',
            max_features=n_components * 0.3,
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
        if complexity == 1:
            return LogisticRegression(
                penalty='l2',
                solver='liblinear',
                random_state=ModelConstructor.RANDOM_STATE
            )
        elif complexity == 2:
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
                random_state=ModelConstructor.RANDOM_STATE
            )
        elif kernel == 'poly':
            return SVC(
                kernel='poly',
                degree=3,
                gamma='scale',
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
    def get_nn_model(granularity, n_components, version):
        if granularity.startswith('per-disk'):
            # Eventually add logic here to say determine if PCA
            # was done or not and respond accordingly. But for now,
            # just assume PCA was done and handle different versions.
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
                                                    use_bias=False, 
                                                    kernel_initializer='he_normal')(previous_layer)
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

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            model.summary()
            return model
        elif granularity == 'per-blob':
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
                                                    use_bias=False, 
                                                    kernel_initializer='he_normal')(previous_layer)
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

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            model.summary()
            return model
