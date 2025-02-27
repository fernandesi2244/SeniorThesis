from CoronalDataLoader import SEPInputDataGenerator
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import multiprocessing
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

NAME = 'sep_prediction_coronal_data_v1'

def build_model():
    """
    Input format per batch sample (all as flattened scalar values immediately following each other)
    - BLOB_ONE_TIME_INFO (9 elements)
    - TIMESERIES_STEPS * len(BLOB_VECTOR_COLUMNS_GENERAL) (27 * 6 = 162 elements)

    Make an array for each of the timeseries semantic vectors and do 1D time series
    convolution on them.
    Combine the timeseries convolution results with the BLOB_ONE_TIME_INFO with a dense layer.
    Output a single value with a sigmoid activation function representing the probability
    of an SEP event occurring within 24 hours of the current time step.
    """

    complete_input_size = (len(SEPInputDataGenerator.BLOB_ONE_TIME_INFO) +
                           SEPInputDataGenerator.TIMESERIES_STEPS * len(SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL))

    flattened_input = tf.keras.layers.Input(shape=(complete_input_size,))

    # Split the input into the three main components

    # One-time info
    one_time_info_input = tf.keras.layers.Lambda(lambda x: x[:, 0:len(SEPInputDataGenerator.BLOB_ONE_TIME_INFO)])(flattened_input)

    # Time-series data
    start_idx = len(SEPInputDataGenerator.BLOB_ONE_TIME_INFO)
    blob_vector_input = tf.keras.layers.Lambda(lambda x: 
        tf.reshape(x[:, start_idx:], (-1, SEPInputDataGenerator.TIMESERIES_STEPS, len(SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL)))
    )(flattened_input)

    # Process the timeseries data
    # TODO: Try smaller kernel size and different number of layers
    timeseries_conv = tf.keras.layers.Conv1D(16, 3, activation='relu')(blob_vector_input)
    timeseries_conv = tf.keras.layers.BatchNormalization()(timeseries_conv)
    timeseries_conv = tf.keras.layers.Conv1D(32, 2, activation='relu')(timeseries_conv)
    timeseries_conv = tf.keras.layers.BatchNormalization()(timeseries_conv)
    timeseries_conv = tf.keras.layers.Flatten()(timeseries_conv)

    # Combine the one-time info and timeseries data with a dense layer
    combined_data = tf.keras.layers.concatenate([one_time_info_input, timeseries_conv])
    combined_data = tf.keras.layers.Dense(64, activation='relu')(combined_data)
    combined_data = tf.keras.layers.BatchNormalization()(combined_data)
    combined_data = tf.keras.layers.Dropout(0.2)(combined_data) # TODO: Play around with more dropout layers

    # Output layer
    output = tf.keras.layers.Dense(1, activation='sigmoid')(combined_data)
    # Prevents log(0) issue
    epsilon = 1e-10
    output = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, epsilon, 1 - epsilon))(output)

    model = tf.keras.Model(inputs=flattened_input, outputs=output)

    model.compile(
                  optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
                 )

    # TODO: print out model architecture and parameter count stats
    model.summary()

    return model

blob_df_filename = '../OutputData/UnifiedActiveRegionData_with_all_events_including_new_flares_and_TEBBS_fix.csv'
blob_df = pd.read_csv(blob_df_filename)
batch_size = 32
shuffle = True

blob_df['Produced an SEP'] = (blob_df['Number of SEPs Produced'] > 0) * 1 # 1 if produced, 0 O.W.
blob_df['Year'] = blob_df['Filename General'].apply(lambda x: x.split('.')[3][0:4])
blob_df['Is Plage'] = blob_df['Is Plage'].astype(int)
"""
'Relevant Active Regions' format is either:
- '[13267]'
- '[13268, 13267]'
- '['9276-2']'

The most probable AR number for the above are:
- '13267'
- '13268' (the first num in the string list)
- '9276-2' (the string as is)

"""
blob_df['Most Probable AR Num'] = blob_df['Relevant Active Regions'].apply(lambda x: x.strip("[]'").split(',')[0])

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

# TODO: Try it making it so that all SEPs from one region are exclusively in one set to prevent leakage.

years = blob_df['Year'].unique()
train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()

for year in years:
    blobs_in_year = blob_df[blob_df['Year'] == year]
    print('Year:', year, 'Number of blobs:', len(blobs_in_year))
    print('Number of SEPs:', blobs_in_year['Produced an SEP'].sum())

    """"""
    # Determine whether each activeRegionNum has produced an SEP at least once
    grouped = blobs_in_year.groupby('Most Probable AR Num')['Produced an SEP'].max()  # max() checks if any record has SEP=1

    # If there is only one active region for which an SEP has been produced, include it in the training set and do a random split on the rest of the dataset from that year
    # This is because the train_test_split had a special bad case for for less than 2 samples of a class
    if grouped[grouped == 1].count() == 1:
        min_train_regions = grouped[grouped == 1].index
        # split the rest randomly between train and test
        remaining_train_regions, test_regions = train_test_split(
            grouped[grouped == 0].index, test_size=0.2, random_state=42
        )
        train_regions = np.concatenate([min_train_regions, remaining_train_regions])
    else:
        # Split active region groups into train and test while preserving class balance
        train_regions, test_regions = train_test_split(
            grouped.index, test_size=0.2, stratify=grouped, random_state=42
        )

    # Select records based on their active region
    train_from_year = blobs_in_year[blobs_in_year['Most Probable AR Num'].isin(train_regions)]
    test_from_year = blobs_in_year[blobs_in_year['Most Probable AR Num'].isin(test_regions)]

    # Stratify again within the train set to create a validation set
    grouped_train = train_from_year.groupby('Most Probable AR Num')['Produced an SEP'].max()
    #print('grouped_train for train/val split:', grouped_train)
    #print('where produced SEP:', grouped_train[grouped_train == 1].count())
    if grouped_train[grouped_train == 1].count() == 1:
        # If there is only one active region for which an SEP has been produced, include it in the training set and do a random split on the rest of the dataset from that year
        min_train_regions = grouped_train[grouped_train == 1].index
        # split the rest randomly between train and validation
        remaining_train_regions, val_regions = train_test_split(
            grouped_train[grouped_train == 0].index, test_size=0.25, random_state=42
        )
        train_regions = np.concatenate([min_train_regions, remaining_train_regions])
    else:
        train_regions, val_regions = train_test_split(
            grouped_train.index, test_size=0.25, stratify=grouped_train, random_state=42
        )

    #print('train_regions:', train_regions)
    #print('val_regions:', val_regions)

    # Select records for train and validation
    new_train_from_year = train_from_year[train_from_year['Most Probable AR Num'].isin(train_regions)]
    val_from_year = train_from_year[train_from_year['Most Probable AR Num'].isin(val_regions)]

    # Now, train, validation, and test sets contain entire active regions with minimal leakage

    # print the regions where 'Produced an SEP' is 1 for both train and test
    print('Len train set:', len(new_train_from_year))
    print('Len val set:', len(val_from_year))
    print('Len test set:', len(test_from_year))

    print('Train regions with SEPs:', new_train_from_year[train_from_year['Produced an SEP'] == 1]['Most Probable AR Num'].unique())
    print('Val regions with SEPs:', val_from_year[val_from_year['Produced an SEP'] == 1]['Most Probable AR Num'].unique())
    print('Test regions with SEPs:', test_from_year[test_from_year['Produced an SEP'] == 1]['Most Probable AR Num'].unique())

    """"""

    # train_from_year, test_from_year = train_test_split(blobs_in_year, test_size=0.2, stratify=blobs_in_year['Produced an SEP'])
    # train_from_year, val_from_year = train_test_split(train_from_year, test_size=0.25, stratify=train_from_year['Produced an SEP'])

    if len(new_train_from_year) == 0 or len(val_from_year) == 0 or len(test_from_year) == 0:
        print('This is literally impossible.')
        print('Year:', year, 'has no data in one of the sets. Skipping.')

    train_df = pd.concat([train_df, new_train_from_year])
    val_df = pd.concat([val_df, val_from_year])
    test_df = pd.concat([test_df, test_from_year])

scaler = StandardScaler()
cols_to_scale = SEPInputDataGenerator.BLOB_VECTOR_COLUMNS_GENERAL + SEPInputDataGenerator.BLOB_ONE_TIME_INFO
train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
val_df[cols_to_scale] = scaler.transform(val_df[cols_to_scale])
test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

# Randomly oversample the majority class and undersample the minority class in the training set for a better balance.
# Use RandomOversampler w/a sampling strategy of 0.325 and then use RandomUndersampler w/a sampling strategy of 0.65.
# These "more optimal" ratios were determined from the other NN models from junior year research.
print('Before resampling:')
print('Train set count:', len(train_df))
print('Train set SEP count:', train_df['Produced an SEP'].sum())

# NOTE: ~37.5x increase in number of SEPs in train set - is quite a huge increase, consider reducing the ratio
ros = RandomOverSampler(sampling_strategy=0.325)
train_df, _ = ros.fit_resample(train_df, train_df['Produced an SEP'])

rus = RandomUnderSampler(sampling_strategy=0.65)
train_df, _ = rus.fit_resample(train_df, train_df['Produced an SEP'])

print('After resampling:')
print('Train set count:', len(train_df))
print('Train set SEP count:', train_df['Produced an SEP'].sum())

# Print the number of SEP-producing blobs in each set
print('Train set SEP count:', train_df['Produced an SEP'].sum())
print('Val set SEP count:', val_df['Produced an SEP'].sum())
print('Test set SEP count:', test_df['Produced an SEP'].sum())

# Use the same random seed so that in case we do shuffling with a random seed in the
# generator, we get the same shuffling for each set.
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

cpus_to_use = max(int(multiprocessing.cpu_count() * 0.9), 1)
print('Using', cpus_to_use, 'CPUs.')

# Quick NaN check
print('Quick train_df check before training')
print(train_df.isna().sum())  # Check for NaNs
print(np.isinf(train_df).sum())  # Check for Infs
print(train_df.describe())

print('Quick val_df check before training')
print(val_df.isna().sum())  # Check for NaNs
print(np.isinf(val_df).sum())  # Check for Infs
print(val_df.describe())

exit()

train_generator = SEPInputDataGenerator(train_df, batch_size, True, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
val_generator = SEPInputDataGenerator(val_df, batch_size, True, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)
test_generator = SEPInputDataGenerator(test_df, batch_size, True, use_multiprocessing=True, workers=cpus_to_use, max_queue_size=cpus_to_use * 2)

print('Number of batches in train:', len(train_generator))
print('Number of batches in val:', len(val_generator))
print('Number of batches in test:', len(test_generator))

model = build_model()

# Define some callbacks to improve training.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)
checkpoint_best_every_20 = tf.keras.callbacks.ModelCheckpoint(
    f"{NAME}_checkpoint_best_every_20.keras",
    save_best_only=True,
    save_freq=20, # save every 20 batches  
)
checkpoint_every_20 = tf.keras.callbacks.ModelCheckpoint(
    f"{NAME}_checkpoint_every_20.keras",
    save_best_only=False,
    save_freq=20, # save every 20 batches  
)

# Start timer
start = time.time()

model.fit(train_generator, epochs=10, validation_data=val_generator,
            callbacks=[early_stopping, reduce_lr, checkpoint_best_every_20, checkpoint_every_20])

# Print the time that has elapsed during training
print('Training time:', time.time() - start)

# Evaluate the model using the val data
val_loss = model.evaluate(val_generator)
print('Val loss:', val_loss)

# Save the model
model.save(f'{NAME}_model.keras')
