from FullDataLoader import SEPInputDataGenerator
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import multiprocessing

def build_model():
    """
    Input format per batch sample, where each bullet point is a numpy array in the overall numpy array:
    - len(BLOB_ONE_TIME_INFO) = 5
    - (TIMESERIES_STEPS, len(BLOB_VECTOR_COLUMNS_GENERAL)) = 27 * 6 = 162 elements total
    - 200 x 400 x 100 x 3 (bx_3D, by_3D, bz_3D)

    Do timeseries convolution on the TIMESERIES_STEPS arrays following the intial array.
    Perform 3D convolution on the 3D magnetic field stacked array.
    Combine the timeseries convolution results with the BLOB_ONE_TIME_INFO and
    3D magnetic field outputs with a dense layer.
    Output a single value with a sigmoid activation function representing the probability
    of an SEP event occurring within 24 hours of the current time step.
    """

    one_time_info_input = tf.keras.layers.Input(shape=(len(SEPInputDataGenerator.BLOB_ONE_TIME_INFO),))
    blob_vector_input = tf.keras.layers.Input(shape=(SEPInputDataGenerator.TIMESERIES_STEPS, len(SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL)))
    volume_blob_input = tf.keras.layers.Input(shape=(200, 400, 100, 3))

    # Process the timeseries data
    # TODO: Try smaller kernel size and different number of layers
    timeseries_conv = tf.keras.layers.Conv1D(16, 3, activation='relu')(blob_vector_input)
    timeseries_conv = tf.keras.layers.Conv1D(32, 2, activation='relu')(timeseries_conv)
    timeseries_conv = tf.keras.layers.Flatten()(timeseries_conv)

    # TODO: Try processing magnitudes
    # TODO: Should we just have each 3D magnetic field component as a separate channel?
    # TODO: Try different strategies for detecting larger-scale structures in the 3D magnetic field data.

    # Perform convolution on the volume_blob_input
    volume_blob_conv = tf.keras.layers.Conv3D(32, 3, activation='relu')(volume_blob_input)
    volume_blob_conv = tf.keras.layers.MaxPooling3D(2)(volume_blob_conv)
    volume_blob_conv = tf.keras.layers.Conv3D(64, 3, activation='relu')(volume_blob_conv)
    volume_blob_conv = tf.keras.layers.MaxPooling3D(2)(volume_blob_conv)
    volume_blob_conv = tf.keras.layers.Conv3D(128, 3, activation='relu')(volume_blob_conv)
    volume_blob_conv = tf.keras.layers.MaxPooling3D(2)(volume_blob_conv)
    volume_blob_conv = tf.keras.layers.Flatten()(volume_blob_conv)

    # Combine the one-time info, timeseries, and 3D magnetic field component data with a dense layer
    combined_data = tf.keras.layers.concatenate([one_time_info_input, timeseries_conv, volume_blob_conv])
    combined_data = tf.keras.layers.Dense(256, activation='relu')(combined_data)
    combined_data = tf.keras.layers.Dense(128, activation='relu')(combined_data)
    combined_data = tf.keras.layers.Dense(64, activation='relu')(combined_data)

    # Output layer
    output = tf.keras.layers.Dense(1, activation='sigmoid')(combined_data)

    model = tf.keras.Model(inputs=[one_time_info_input, blob_vector_input, volume_blob_input], outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # TODO: print out model architecture and parameter count stats
    model.summary()

    return model

blob_df_filename = '../OutputData/UnifiedActiveRegionData.csv'
blob_df = pd.read_csv(blob_df_filename)
batch_size = 5
shuffle = True

blob_df['Produced an SEP'] = (blob_df['Number of SEPs Produced'] > 0) * 1 # 1 if produced, 0 O.W.
blob_df['Year'] = blob_df['Filename General'].apply(lambda x: x.split('.')[3][0:4])
blob_df['Is Plage'] = blob_df['Is Plage'].astype(int)

# TODO: Remove this
print(blob_df['Is Plage'])
exit()

# Just for overall batch count
generator = SEPInputDataGenerator(blob_df, batch_size, True)
print('Number of batches overall:', len(generator))
print('Length of each batch:', batch_size)

# For each year of data, split it into 3 parts: 60% training, 20% validation, 20% test.
# Stratify the split so the class imbalance is preserved.
# To form the complete training, validation, and test sets, concatenate the splits for each year.
# Randomly shuffle each set. Make sure to use a random seed for all shuffling/random operations.
# NOTE: there will be some small data leakage between the training, validation, and test sets because the
# a small subset of active regions will appear in multiple sets. Perhaps explore some margin between
# the training, validation, and test sets to reduce this leakage.

years = blob_df['Year'].unique()
train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()

for year in years:
    blobs_in_year = blob_df[blob_df['Year'] == year]

    train_from_year, test_from_year = train_test_split(blobs_in_year, test_size=0.2, stratify=blobs_in_year['Produced an SEP'])
    train_from_year, val_from_year = train_test_split(train_from_year, test_size=0.25, stratify=train_from_year['Produced an SEP'])

    train_df = pd.concat([train_df, train_from_year])
    val_df = pd.concat([val_df, val_from_year])
    test_df = pd.concat([test_df, test_from_year])

scaler = StandardScaler()
cols_to_scale = SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL + SEPInputDataGenerator.BLOB_ONE_TIME_INFO
train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
val_df[cols_to_scale] = scaler.transform(val_df[cols_to_scale])
test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

# Use the same random seed so that in case we do shuffling with a random seed in the
# generator, we get the same shuffling for each set.
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_generator = SEPInputDataGenerator(train_df, batch_size, True)
val_generator = SEPInputDataGenerator(val_df, batch_size, True)
test_generator = SEPInputDataGenerator(test_df, batch_size, True)

print('Number of batches in train:', len(train_generator))
print('Number of batches in val:', len(val_generator))
print('Number of batches in test:', len(test_generator))

model = build_model()

# Define some callbacks to improve training.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
checkpoint = tf.keras.callbacks.ModelCheckpoint("sep_prediction_v1_checkpoint.keras", save_best_only=True)

# Start timer
start = time.time()

cpus_to_use = max(int(multiprocessing.cpu_count() * 0.9), 1)
print('Using', cpus_to_use, 'CPUs.')

model.fit(train_generator, epochs=10, validation_data=val_generator,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2 # TODO: Play around with this
            )

# Print the time that has elapsed during training
print('Training time:', time.time() - start)

# Evaluate the model using the val data
val_loss = model.evaluate(val_generator)
print('Val loss:', val_loss)

# Save the model
model.save('sep_prediction_v1_model.keras')
