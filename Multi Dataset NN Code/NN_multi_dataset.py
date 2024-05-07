# Author: Nirosan Pragash
# Master's Dissertation for MEng Integrated Mechanical and Electrical Engineering
# University of Bath
#
# This script is part of a research project that involves constructing and tuning a deep learning model
# for predicting battery performance characteristics. The model is built using Keras and TensorFlow,
# and is optimised using the Keras Tuner library. The data used for training and testing are derived from
# multiple datasets, including CALCE, Oxford, and CAS. Each dataset provides voltage and charge curves
# Using the optimal parameters found through NN_multi_dataset_hyperparameter_tuner.py, this code trains and
# tests the model.

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime
from keras import optimizers
from keras import layers
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import custom_object_scope
import glob
from keras import backend as K
import random

train = 0 # Toggles the training of the model
showplot = 0 # Toggles plots
whosmodel = 'multi' # Selects the model to train and test: multi or original

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

# Reducing the size of datasets by a factor for computational effiency.
cutdownratio = 5
curve_cell_1 = curve_cell_1[0:-1:cutdownratio]  
curve_cell_2 = curve_cell_2[0:-1:cutdownratio]  
curve_cell_3 = curve_cell_3[0:-1:cutdownratio]  
curve_cell_4 = curve_cell_4[0:-1:cutdownratio]  
curve_cell_5 = curve_cell_5[0:-1:cutdownratio]  
curve_cell_6 = curve_cell_6[0:-1:cutdownratio]  
curve_cell_7 = curve_cell_7[0:-1:cutdownratio]  
curve_cell_8 = curve_cell_8[0:-1:cutdownratio]  
curve_cell_9 = CAS_data[0:-1:cutdownratio]   # Divides the CAS charge cycles into two - test data and as 
curve_cell_10 = CAS_data[1:-1:cutdownratio]  # training data

# Scaling the capacity curves of Oxford to match that of CALCE.
curve_cell_5 = (curve_cell_5/0.74)*1.1
curve_cell_6 = (curve_cell_6/0.74)*1.1
curve_cell_7 = (curve_cell_7/0.74)*1.1
curve_cell_8 = (curve_cell_8/0.74)*1.1

# Scaling the capacity curves of CAS to match that of CALCE.
CAS_data = (CAS_data/11.6)*1.1

# Defining the voltage range.
voltage = np.linspace(3.6, 4.1, 140)

# Diving the charge curves into training and testinf data.
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

# If training is enabled
if train ==1:
    # Setup for TensorBoard
    log_dir = os.path.join('logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    if whosmodel == 'multi':
        # Define the model architecture
        model = Sequential([
            Conv1D(filters=1648, kernel_size=7, activation='elu', padding='causal', input_shape=(None, 2)),
            MaxPooling1D(2),
            Conv1D(filters=720, kernel_size=11, activation='elu', padding='causal'),
            MaxPooling1D(2),
            GlobalMaxPooling1D(),
            Dense(8650, activation='relu'),
            Dropout(0.1),
            Dense(len(voltage))  # Ensure 'voltage' is defined with the actual output dimension.
        ])

        # Set the learning rate and optimizer
        adam_optimizer = Adam(learning_rate=0.00010591, clipvalue=0.99999)

        # Compile the model with the updated optimizer
        model.compile(loss=custom_loss, optimizer=adam_optimizer, metrics=['accuracy'])

    if whosmodel == 'original':
        # Original model architecture
        model = Sequential()
        model.add(layers.Conv1D(16, 3, activation='relu', padding='causal', input_shape=(None, Input_train.shape[-1])))
        model.add(layers.MaxPooling1D(3))
        model.add(layers.Conv1D(8, 3, activation='relu', padding='causal'))
        model.add(layers.MaxPooling1D(3))
        model.add(layers.Conv1D(8, 3, activation='relu', padding='causal'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(140, activation='relu'))
        model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(len(voltage)))
        model.summary()

        # Optimiser setup
        optim = optimizers.Adam()
        
        # Compiling the model
        model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])



    # Set up TensorBoard for visualization
    log_dir = os.path.join('logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    if whosmodel == 'multi':
        filepath = "models_multi/{epoch:02d}-{val_loss:.2f}.hdf5"
    else:   
        filepath = "models_original/{epoch:02d}-{val_loss:.2f}.hdf5"

    # ModelCheckpoint callback to save the best model
    model_checkpoint_callback = ModelCheckpoint(
        filepath=filepath,  # Path to save the models.
        monitor='val_loss',  # Monitor the validation loss
        verbose=1,  # Log the saving of the model
        save_best_only=True,  # Save only the best model according to the monitored metric
        mode='auto'  # The decision to overwrite is made automatically based on the validation loss
    )

    # Train the model
    history = model.fit(
        Input_train,  # Training input data
        Output_train,  # Training output data
        epochs=5000,  # Number of epochs for training
        validation_split=0.4,  # Percentage of data used for validation
        callbacks=[tensorboard_callback, model_checkpoint_callback],  # Callbacks for TensorBoard and ModelCheckpoint
        verbose=2  # Set to True to see progress bar for each epoch
    )
    # Print the summary of the model architecture
    model.summary()

    # Calculate and print the elapsed time
    print("--- %s seconds ---" % (time.time() - start_time))

# Directory containing the model files
if whosmodel == 'multi':
    model_dir = '/models_multi'
else:
    model_dir = '/models_original'

# List all .hdf5 files in the model directory
hdf5_files = glob.glob(os.path.join(model_dir, '*.hdf5'))

# Sort files by modification date, newest first
hdf5_files.sort(key=os.path.getmtime, reverse=True)

# Select the newest file
newest_model_file = hdf5_files[0]

# Load the newest model with the custom loss
if whosmodel == 'multi':
    with custom_object_scope({'custom_loss': custom_loss}):
        model = load_model(newest_model_file)
else:
    model = load_model(newest_model_file)

# Print a message indicating that the model has been loaded

# Calculate and print the elapsed time
print("--- %s seconds ---" % (time.time() - start_time))


# Make predictions for the test dataset
predicted_test = model.predict(Input_test)

if showplot == 1:
    num_plots = 15

    # Plots the predicted and actual curve for the first 20 samples.
    # Create a figure and axes with a grid of subplot.
    fig, axs = plt.subplots(3, 5, figsize=(15, 10))  # Adjusted to a 3x5 grid layout
    # Flatten the array of axes to make it easier to iterate over
    axs = axs.flatten()
    # Use a counter for the axes index
    axes_counter = 0

    # Iterate over the range and plot all cycles
    for i in range(1, num_plots + 1):
        # Plot predicted data
        axs[axes_counter].plot(voltage, predicted_test[i-1, :], 'r-', label='Predicted')  # Adjust index to i-1
        
        # Plot actual data
        axs[axes_counter].plot(voltage, Output_test[i-1, :], 'b--', label='Actual')  # Adjust index to i-1
        
        # Add some plot decorations
        axs[axes_counter].set_xlabel('Voltage (V)')
        axs[axes_counter].set_ylabel('Charge Capacity (Ah)')
        axs[axes_counter].legend()
        
        # Increment the axes counter
        axes_counter += 1

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Show the plot
    plt.show()


    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Plots the predicted and actual curve for 20 randomly selected samples.
    total_cycles = predicted_test.shape[0]

    # Generate a list of random indices from the total available cycles
    random_indices = random.sample(range(total_cycles), num_plots)

    # Create a figure and axes with a grid of subplots
    fig, axs = plt.subplots(3, 5, figsize=(15, 10))  # Adjusted to a 3x5 grid layout

    # Flatten the array of axes to make it easier to iterate over
    axs = axs.flatten()

    # Plot each of the randomly selected cycles' predicted and actual data
    for i, idx in enumerate(random_indices):
        # Plot predicted data
        axs[i].plot(voltage, predicted_test[idx, :], 'r-', label='Predicted')  # Solid red line for predicted
        
        # Plot actual data
        axs[i].plot(voltage, Output_test[idx, :], 'b--', label='Actual')  # Dashed blue line for actual
        
        # Add some plot decorations
        axs[i].set_xlabel('Voltage (V)')
        axs[i].set_ylabel('Charge Capacity (Ah)')
        axs[i].legend()

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Show the plot
    plt.show()

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Plots the predicted and actual curve for the last 20 samples. 
    # Select the last 20 indices
    last_indices = range(len(Output_test) - 20, len(Output_test))

    # Create a figure and axes with a grid of subplots
    fig, axs = plt.subplots(3, 5, figsize=(15, 10))  # Adjusted to a 3x5 grid layout

    # Iterate over each subplot
    for i, ax in enumerate(axs.flat):
        idx = last_indices[i]
        
        # Plot actual data for this index
        ax.plot(voltage, Output_test[idx], 'b--', label='Actual')  # Dashed blue line for actual
        
        # Plot predicted data for this index
        ax.plot(voltage, predicted_test[idx], 'r-', label='Predicted')  # Solid red line for predicted
        
        # Set the title, labels, and legend
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Charge Capacity (Ah)')
        ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

    #////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    #Plots the predicted and actual curve for 16 randomly selected samples with the section used to predict the curve 
    # highlighted.
    plt.figure(figsize=(15, 10))

    # Selecting random indices
    num_plots = 16
    selected_indices = random.sample(range(len(Output_test)), num_plots)

    # Find the maximum value of Output_test among the selected indices
    max_value = np.max([np.max(Output_test[i]) for i in selected_indices])

    for idx, i in enumerate(selected_indices, start=1):
        plt.subplot(4, 4, idx)
        
        # Plotting the actual values
        plt.plot(voltage, Output_test[i], 'b--', label='Actual')  # Dashed blue line
        
        # Plotting the predicted values
        plt.plot(voltage, predicted_test[i], 'r-', label='Predicted')  # Solid red line
        
        # Plotting the lookback window data
        lookback_window_voltage = Input_test[i, :, 0] * std[0] + mean[0]
        lookback_window_capacity = Input_test[i, :, 1] * std[1] + mean[1]

        # Create a polygon to fill the area under the curve
        x_polygon = np.concatenate([lookback_window_voltage, lookback_window_voltage[::-1]])
        y_polygon = np.concatenate([np.zeros_like(lookback_window_capacity), np.full_like(lookback_window_capacity, max_value)])
        plt.fill(x_polygon, y_polygon, color='lightgreen', alpha=0.3)  # Shade the area

        plt.xlabel('Voltage (V)')
        plt.ylabel('Capacity')
        plt.legend()

    plt.tight_layout()
    plt.show()