# Author: Nirosan Pragash
# Master's Dissertation for MEng Integrated Mechanical and Electrical Engineering
# University of Bath
#
# This script is part of a research project that involves constructing and tuning a deep learning model
# for predicting battery performance characteristics. The model is built using Keras and TensorFlow,
# and is optimised using the Keras Tuner library. The data used for training and testing are derived from
# multiple datasets, including CALCE, Oxford, and CAS. Each dataset provides voltage and charge curves
# for various battery cells, which are then pre-processed and fed into the model.
# This code fine-tunes the trial hyperparameters iteratively to find the most optimal hyperparameters for the task.

import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
import keras_tuner as kt

# Record the start time of the script
start_time = time.time()

# Paths to the directories containing the datasets. Ensure these are correctly set to the dataset locations on your system.
location_CALCE = ''
location_Oxford = ''
location_CAS = ''

# Load data: Importing battery charge curves from specified locations for different datasets.
# Each dataset has its own characteristics, and the curves represent different battery behaviours under various conditions.
curve_cell_1 = np.genfromtxt(location_CALCE+'CS2_35.txt', delimiter=',')
curve_cell_2 = np.genfromtxt(location_CALCE+'CS2_36.txt', delimiter=',')
curve_cell_3 = np.genfromtxt(location_CALCE+'CS2_37.txt', delimiter=',')
curve_cell_4 = np.genfromtxt(location_CALCE+'CS2_38.txt', delimiter=',')
curve_cell_5 = np.genfromtxt(location_Oxford+'Cell1.txt', delimiter=',')
curve_cell_6 = np.genfromtxt(location_Oxford+'Cell3.txt', delimiter=',')
curve_cell_7 = np.genfromtxt(location_Oxford+'Cell7.txt', delimiter=',')
curve_cell_8 = np.genfromtxt(location_Oxford+'Cell8.txt', delimiter=',')
CAS_data = np.genfromtxt(location_CAS+'batt07_cccv.txt', delimiter=',')

# Preprocessing: Adjusting data to ensure uniformity in scale and magnitude across different datasets.
# This step is critical for the training process, as it normalises data inputs to aid in model convergence.

# Reducing the size of datasets by a factor for effiency.
cutdownratio = 5
curve_cell_1 = curve_cell_1[0:-1:cutdownratio]  
curve_cell_2 = curve_cell_2[0:-1:cutdownratio]  
curve_cell_3 = curve_cell_3[0:-1:cutdownratio]  
curve_cell_4 = curve_cell_4[0:-1:cutdownratio]  
curve_cell_5 = curve_cell_5[0:-1:cutdownratio]  
curve_cell_6 = curve_cell_6[0:-1:cutdownratio]  
curve_cell_7 = curve_cell_7[0:-1:cutdownratio]  
curve_cell_8 = curve_cell_8[0:-1:cutdownratio]  
curve_cell_9 = CAS_data[0:-1:cutdownratio]   
curve_cell_10 = CAS_data[1:-1:cutdownratio]

# Scaling the capacity curves of Oxford to match that of CALCE.
curve_cell_5 = (curve_cell_5/0.74)*1.1
curve_cell_6 = (curve_cell_6/0.74)*1.1
curve_cell_7 = (curve_cell_7/0.74)*1.1
curve_cell_8 = (curve_cell_8/0.74)*1.1

# Scaling the capacity curves of CAS to match that of CALCE.
CAS_data = (CAS_data/11.6)*1.1

# Defining the voltage range.
voltage = np.linspace(3.6, 4.1, 140)

# Diving the charge curves into training and testing data.
curve_train = [curve_cell_2, curve_cell_6, curve_cell_9]
curve_test = [curve_cell_1, curve_cell_3, curve_cell_4, curve_cell_5, curve_cell_7, curve_cell_8, curve_cell_10] 

# Flatten and combine charge data from curve_train
entire_charge = curve_train[0].flatten()
for ind in range(1, len(curve_train)):
    entire_charge = np.append(entire_charge, curve_train[ind].flatten())

# Replicate voltage data to match the length of entire_charge
entire_voltage = np.tile(voltage, len(entire_charge) // len(voltage))

# Stack charge and voltage data vertically and transpose
entire_series_stack = np.vstack((entire_voltage, entire_charge))
entire_series = entire_series_stack.T

# Print shapes of charge, voltage, and combined series
print(entire_charge.shape)
print(entire_voltage.shape)
print(entire_series.shape)

# Calculate mean and standard deviation of the training data for normalisation
mean = entire_series.mean(axis=0)
entire_series -= mean
std = entire_series.std(axis=0)
entire_series /= std

def generator(data, lookback, delay, min_index, max_index,
                            shuffle=False, batch_size=128, step=1):
    """
    Generates training and testing data sequences.

    Args:
    - data: The input data.
    - lookback: The number of timesteps the input data should go back.
    - delay: The number of timesteps the target should be in the future.
    - min_index: The minimum index in the data array.
    - max_index: The maximum index in the data array.
    - shuffle: Whether to shuffle the data or not.
    - batch_size: The batch size.
    - step: The sampling rate of the original data.

    Returns:
    - samples: The generated data sequences.
    """
    if max_index is None:
        max_index = len(data) - delay - 1  
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                            lookback // step - step,
                            data.shape[-1]))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices][1:, :]
            samples[j][:, 1] -= data[indices][0, 1]
        return samples

# Training Data Preparation
# This section prepares training data by normalising and generating samples through a sliding window technique.
# Initialize lists for temporary storage of training data and targets.
data_train_temp = []
target_train_temp = []

# Loop through each dataset in 'curve_train' to process individual charge curves.
for ind in range(len(curve_train)):
    for k in range(len(curve_train[ind])):
        charge = curve_train[ind][k]  # Extract charge data for the current curve.

        # Stack voltage and charge arrays, transpose to align columns as features.
        temp_train_vstack = np.vstack((voltage, charge)).T  

        # Normalize the data: zero-center and scale.
        temp_train = (temp_train_vstack - mean) / std

        # Define batch size equal to the length of data for full sequence processing.
        batch_size_train = len(temp_train)

        # Generate normalized training samples using a custom generator function.
        train_gen = generator(temp_train,
                                            lookback=31,  # Define the window size for each sequence.
                                            delay=0,      # No delay between input and output pairs.
                                            min_index=0,  # Starting index in the dataset.
                                            max_index=None,  # Process up to the last index.
                                            shuffle=True,  # Shuffle sequences to reduce model overfitting.
                                            batch_size=batch_size_train,
                                            step=1)

        # Store generated samples and replicate charge data for training targets.
        data_train_temp.append(train_gen)
        A = np.tile(charge, [len(train_gen), 1])
        target_train_temp.append(A)

# Concatenate all training data and targets into final arrays.
train_gen_final = np.concatenate(data_train_temp, axis=0)
train_target_final = np.concatenate(target_train_temp, axis=0)

# Test Data Preparation 
data_test_temp = []
target_test_temp = []

# Loop through each dataset in 'curve_test' to process individual charge curves.
for ind in range(len(curve_test)):
    for k in range(len(curve_test[ind])):
        charge = curve_test[ind][k]

        # Stack voltage and charge arrays, transpose to align columns as features.
        temp_test_vstack = np.vstack((voltage, charge))
        temp_test_not = temp_test_vstack.T

        # Standardization (without re-normalization)
        temp_test = (temp_test_not - mean) / std

        # Define batch size equal to the length of data for full sequence processing.
        batch_size_test = len(temp_test)

        # Generate test samples using a custom generator function.
        test_gen = generator(temp_test,
                                            lookback=31,
                                            delay=0,
                                            min_index=0,
                                            max_index=None,
                                            shuffle=True,
                                            batch_size=batch_size_test,
                                            step=1)
        
        # Store generated samples and replicate charge data for testing targets.
        data_test_temp.append(test_gen)
        A = np.tile(charge, [len(test_gen), 1])
        target_test_temp.append(A)

# Concatenate all test data and targets into final arrays.
test_gen_final = np.concatenate(data_test_temp, axis=0)
test_target_final = np.concatenate(target_test_temp, axis=0)

# Print shapes of final test data arrays for verification.
print(test_gen_final.shape)
print(test_target_final.shape)

# Shuffle the training dataset for validation.
index = np.arange(train_gen_final.shape[0])
np.random.shuffle(index)

# Shuffle the training data and corresponding targets.
Input_train = train_gen_final[index, :, :]
Output_train = train_target_final[index, :]

# Assign test data and targets directly.
Input_test = test_gen_final
Output_test = test_target_final

# Setup for TensorBoard
log_dir = os.path.join('logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


def custom_loss(y_true, y_pred):
    """
    Custom loss function that calculates the mean of weighted squared errors.

    Args:
    - y_true: The true values.
    - y_pred: The predicted values.

    Returns:
    - The mean of the weighted squared errors.
    """
    # Parameters for linear weight increase
    base_weight = 1.0  # The starting weight
    weight_increase_rate = 9.0  # How quickly the weight increases with y_true
    
    # Linearly increasing weights
    weights = base_weight + (weight_increase_rate * y_true)
    
    # Calculate squared error
    squared_error = K.square(y_true - y_pred)

    # Apply weights to the squared error
    weighted_error = squared_error * weights
    
    # Return the mean of the weighted errors
    return K.mean(weighted_error, axis=-1)

def build_model(hp):
    """
    Builds a convolutional neural network model.

    Args:
    - hp: Hyperparameters object.

    Returns:
    - model: A compiled convolutional neural network model.
    """
    # Initialize a sequential model
    model = Sequential([
        # First convolutional layer
        Conv1D(
            filters=hp.Int('conv_1_filters', min_value=16, max_value=2048, step=16),
            kernel_size=hp.Choice('conv_1_kernel_size', values=[3, 5, 7, 9, 11]),
            activation=hp.Choice('conv_1_activation', values=['relu', 'tanh', 'sigmoid', 'softmax', 'elu']),
            padding=hp.Choice('conv_1_padding', values=['same']),  
            input_shape=(None, 2)
        ),
        # First max pooling layer
        MaxPooling1D(2),
        # Second convolutional layer
        Conv1D(
            filters=hp.Int('conv_2_filters', min_value=16, max_value=2048, step=16),
            kernel_size=hp.Choice('conv_2_kernel_size', values=[3, 5, 7, 9, 11]),
            activation=hp.Choice('conv_2_activation', values=['relu', 'elu']),
            padding=hp.Choice('conv_2_padding', values=['same'])  
        ),
        # Second max pooling layer
        MaxPooling1D(2),
        # Global max pooling layer
        GlobalMaxPooling1D(),
        # Dense layer
        Dense(
            hp.Int('dense_units', min_value=50, max_value=25000, step=50),
            activation=hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid', 'softmax', 'elu'])
        ),
        # Dropout layer
        Dropout(hp.Float('dropout', min_value=0.0, max_value=0.9, step=0.01)),
        # Output layer
        Dense(len(voltage)) 
    ])

    # Selecting the optimiser
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    clip_value = 0.99999

    # Set optimizer based on choice
    if optimizer_choice == 'adam':
        optimizer = Adam(
            learning_rate=hp.Float('adam_learning_rate', min_value=1e-4, max_value=0.9, sampling='LOG'),
            clipvalue=clip_value
        )
    elif optimizer_choice == 'sgd':
        optimizer = SGD(
            learning_rate=hp.Float('sgd_learning_rate', min_value=1e-4, max_value=0.9, sampling='LOG'),
            clipvalue=clip_value
        )
    else:
        optimizer = RMSprop(
            learning_rate=hp.Float('rmsprop_learning_rate', min_value=1e-4, max_value=0.9, sampling='LOG'),
            clipvalue=clip_value
        )

    # Compile the model with the custom loss function
    model.compile(optimizer=optimizer, loss=custom_loss)

    return model

# Hyperband tuner initialisation
tuner = kt.Hyperband(
    build_model,  # Function that constructs the model to be tuned
    objective='val_loss',  # Objective to optimize during tuning (minimize validation loss)
    max_epochs=500,  # Maximum number of epochs to train each model configuration
    directory='tuner_results',  # Directory to store the tuning results
    project_name='battery_model_tuning'  # Name of the tuning project
)

# Hyperparameter search
tuner.search(
    Input_train,  # Input training data
    Output_train,  # Target training data
    epochs=500,  # Number of epochs to train each model configuration during the search
    validation_split=0.4,  # Percentage of training data to use for validation
    callbacks=[  # List of callbacks to apply during training
        EarlyStopping(monitor='val_loss', patience=5),  # Stop training if validation loss stops improving
        tensorboard_callback  # TensorBoard callback for visualization
    ],
    verbose=1  # Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch
)

# Retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

# Evaluate the best model on the test dataset
result = model.evaluate(Input_test, Output_test)

# ModelCheckpoint callback to save the model
model_checkpoint_callback = ModelCheckpoint(
    filepath="models/{epoch:02d}-{val_loss:.2f}.hdf5",  # Path to save the models
    monitor='val_loss',  # Save the model based on the validation loss
    verbose=1,  # Log the saving of the model
    save_best_only=True,  # Save only the best model
    mode='auto'  # The decision to overwrite the current save file is made automatically based on the monitoring of the validation loss
)

# Train the model with the fixed hyperparameters and include the ModelCheckpoint in callbacks
history = model.fit(
    Input_train, 
    Output_train, 
    epochs=5000, 
    validation_split=0.4,  # Percentage of data used for validation
    callbacks=[tensorboard_callback, model_checkpoint_callback], 
    verbose=2  # Set to True to see progress bar for each epoch
)

# Print model summary
model.summary()

# Plot training and validation loss
plt.figure(dpi=150)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Extract trial details from the tuner's oracle
trials = tuner.oracle.get_trials()

# Create a list to hold all trial data
all_trials_data = []
for trial_id, trial in tuner.oracle.trials.items():
    # Extract hyperparameters and score for each trial
    trial_data = trial.hyperparameters.values
    trial_data['score'] = trial.score
    all_trials_data.append(trial_data)

# Convert the list to a Pandas DataFrame
trials_dataframe = pd.DataFrame(all_trials_data)

# Export the DataFrame to a CSV file
csv_filename = 'hyperparameter_trials.csv'
trials_dataframe.to_csv(csv_filename, index=False)

# Display confirmation message
print(f'Trial data exported to {csv_filename}.')

# Calculate and display execution time
print(f"--- {time.time() - start_time} seconds ---")