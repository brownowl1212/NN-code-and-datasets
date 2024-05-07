# Author: Nirosan Pragash
# Master's Dissertation Project
# University: University of Bath
# Course: MEng Integrated Mechanical and Electrical Engineering
# This script is designed to process and analyse battery cycling data for different the CALCE dataset.
# It includes data loading, preprocessing, analysis, and visualisation components. Functions
# are provided for plotting cycle data, calculating state of health (SOH), and exporting results.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
import random

# Toggles for exporting data and showing plots
export = 0
showplot = 1

# Base directory in which the CSV files are located
base_dir = ''

# Dictionary mapping series names to their respective CSV file paths
csv_files = {
    'CS2_35': 'CS2_35/CS2_35_combined.csv',
    'CS2_36': 'CS2_36/CS2_36_combined.csv',
    'CS2_37': 'CS2_37/CS2_37_combined.csv',
    'CS2_38': 'CS2_38/CS2_38_combined.csv'
}

# Load CSV files into Pandas DataFrames
CS2_35 = pd.read_csv(f"{base_dir}/{csv_files['CS2_35']}")
CS2_36 = pd.read_csv(f"{base_dir}/{csv_files['CS2_36']}")
CS2_37 = pd.read_csv(f"{base_dir}/{csv_files['CS2_37']}")
CS2_38 = pd.read_csv(f"{base_dir}/{csv_files['CS2_38']}")

# Define columns to retain and filter out data with negative current values
columns_to_keep = ['Cycle_Index', 'Voltage(V)', 'Charge_Capacity(Ah)']
CS2_35 = CS2_35[CS2_35['Current(A)'] > 0][columns_to_keep]
CS2_36 = CS2_36[CS2_36['Current(A)'] > 0][columns_to_keep]
CS2_37 = CS2_37[CS2_37['Current(A)'] > 0][columns_to_keep]
CS2_38 = CS2_38[CS2_38['Current(A)'] > 0][columns_to_keep]

# Consolidate battery data into a list and dictionary for easier processing
batteries = [CS2_35, CS2_36, CS2_37, CS2_38]
battery_names = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
all_charge_data = {}

# Iterate through each battery and its corresponding name in the zip of batteries and battery names
for battery, name in zip(batteries, battery_names):
    cycles = battery['Cycle_Index'].unique()
    battery_cycles_matrices = {}

    for cycle in cycles:
        cycle_data = battery[battery['Cycle_Index'] == cycle][['Charge_Capacity(Ah)', 'Voltage(V)']]
        battery_cycles_matrices[cycle] = cycle_data.to_numpy()

    all_charge_data[name] = battery_cycles_matrices

# Data cleaning for charge cycles
for battery, cycles_data in all_charge_data.items():
    for cycle, cycle_data in cycles_data.items():
        Q_data = cycle_data[:, 0]  
        Voltage_data = cycle_data[:, 1]  

        # Apply data cleaning steps for invalid and out-of-range values
        negative_Q_indices = np.where(Q_data < 0)[0]
        cycle_data = np.delete(cycle_data, negative_Q_indices, axis=0)
        zero_Q_indices = np.where(Q_data == 0)[0]
        cycle_data = np.delete(cycle_data, zero_Q_indices, axis=0)
        voltage_indices = np.where((Voltage_data < 3.6) | (Voltage_data > 4.1))[0]
        cycle_data = np.delete(cycle_data, voltage_indices, axis=0)

        # Update clean data
        all_charge_data[battery][cycle] = cycle_data

# Count cycles post-cleaning
full_cycle_count = {}
for battery, cycles_data in all_charge_data.items():
    full_cycle_count[battery] = len(cycles_data)

# Adjust charge capacities in each cycle
for battery_name, cycles_data in all_charge_data.items():
    for cycle_number, cycle_data in cycles_data.items():
        if cycle_data.size > 0:
            min_q_value = cycle_data[:, 0].min()
            cycle_data[:, 0] -= min_q_value
            all_charge_data[battery_name][cycle_number] = cycle_data

# Define a function to detect sequences of decreasing capacity (Q) within data
def has_decreasing_sequence(capacity_data, sequence_length=2):
    decreasing_count = 0
    for i in range(1, len(capacity_data)):
        if capacity_data[i] < capacity_data[i - 1]:
            decreasing_count += 1
            if decreasing_count >= (sequence_length - 1):
                return True
        else:
            decreasing_count = 0
    return False

# Dictionary to hold batteries and their cycles with decreasing voltage sequences
cycles_with_decreasing_voltage = {}

# Process each battery to identify cycles with decreasing voltage patterns
for battery, cycles_data in all_charge_data.items():
    decreasing_voltage_cycles = []
    for cycle, cycle_data in cycles_data.items():
        voltage_data = cycle_data[:, 1]  
        
        # Check for decreasing voltage sequence
        if has_decreasing_sequence(voltage_data):
            decreasing_voltage_cycles.append(cycle)
    
    # Record batteries with identified decreasing voltage sequences
    if decreasing_voltage_cycles:
        cycles_with_decreasing_voltage[battery] = decreasing_voltage_cycles

# Dictionary to track cycles failing voltage threshold conditions
store_failed_cycle = {}

# Examine each battery for cycles not meeting voltage criteria
for battery, cycles_data in all_charge_data.items():
    failed_cycles = []
    for cycle, cycle_data in cycles_data.items():
        voltage_data = cycle_data[:, 1]  
        
        # Condition check for initial voltage values exceeding 3.65V
        if (len(voltage_data) >= 20 and np.all(voltage_data[:20] > 3.7)) or (len(voltage_data) >= 20 and np.all(voltage_data[-20:] < 4)):
            failed_cycles.append(cycle)
    
    # Record failed cycles
    if failed_cycles:
        store_failed_cycle[battery] = failed_cycles

# Filter and renumber valid cycles
updated_all_charge_data = {}

for battery, cycles_data in all_charge_data.items():
    updated_cycles_data = {}
    valid_cycles_data = {cycle: data for cycle, data in cycles_data.items() if cycle not in store_failed_cycle.get(battery, [])}
    valid_cycles_data = {cycle: data for cycle, data in valid_cycles_data.items() if cycle not in cycles_with_decreasing_voltage.get(battery, [])}
    
    # Renumber remaining cycles
    renumbered_cycle = 1
    for cycle in sorted(valid_cycles_data.keys()):
        updated_cycles_data[renumbered_cycle] = valid_cycles_data[cycle]
        renumbered_cycle += 1
    
    updated_all_charge_data[battery] = updated_cycles_data

all_charge_data.clear()
all_charge_data.update(updated_all_charge_data)


# Function to remove cycles with invalid data points
def remove_cycles_with_invalid_values(charge_data):
    valid_charge_data = {}
    for battery, cycles_data in charge_data.items():
        valid_cycles = {}
        for cycle, cycle_data in cycles_data.items():
            q_data, v_data = cycle_data[:, 0], cycle_data[:, 1]
            if cycle == 1:
                cycle1_max_q = max(q_data)
            is_q_valid = q_data.size >= 50 and not np.any(np.isnan(q_data)) and np.all(np.isfinite(q_data)) and max(q_data) <= cycle1_max_q
            is_v_valid = v_data.size >= 50 and not np.any(np.isnan(v_data)) and np.all(np.isfinite(v_data))
            
            if cycle == 629 and battery == "CS2_35":
                print(q_data)
            
            if is_q_valid and is_v_valid:
                valid_cycles[cycle] = cycle_data
        
        valid_charge_data[battery] = valid_cycles
    return valid_charge_data

all_charge_data = remove_cycles_with_invalid_values(all_charge_data)

# Establish baseline max capacity from the first cycle for each battery
SOH100_capacity = {}
for battery_name, cycles_data in all_charge_data.items():
    first_cycle_data = cycles_data[min(cycles_data, key=int)]
    if first_cycle_data.size > 0:
        SOH100_capacity[battery_name] = first_cycle_data[:, 0].max()

# Update each cycle's data with calculated State of Health (SOH)
for battery_name, cycles_data in all_charge_data.items():
    for cycle_number, cycle_data in sorted(cycles_data.items(), key=lambda x: int(x[0])):
        if cycle_data.size > 0:
            max_capacity = cycle_data[:, 0].max()
            if max_capacity > 0 and battery_name in SOH100_capacity:
                SOH = (max_capacity / SOH100_capacity[battery_name]) * 100
                SOH = int(SOH // 2) * 2  # Round SOH to nearest multiple of 2
                cycle_data_with_SOH = np.column_stack((cycle_data, np.full_like(cycle_data[:, 0], SOH)))
                all_charge_data[battery_name][cycle_number] = cycle_data_with_SOH

# Remove cycles with SOH below the specified threshold
threshold = 64.9
for battery_name in list(all_charge_data.keys()):
    cycles_to_keep = {}
    for cycle_number, cycle_data_with_SOH in all_charge_data[battery_name].items():
        if cycle_data_with_SOH.size > 0 and cycle_data_with_SOH[0, -1] >= threshold:
            cycles_to_keep[cycle_number] = cycle_data_with_SOH
    all_charge_data[battery_name] = cycles_to_keep


# Set the desired count of cycles per State of Health (SOH) value
desired_count = 80

# Process each battery in the dataset
for battery_id in all_charge_data.keys():
    print(f"Processing battery: {battery_id}")
    
    # Identify unique SOH values within the current battery's data
    unique_soh_values = set(data[0, -1] for data in all_charge_data[battery_id].values() if data.size > 0)
    
    # Adjust the number of cycles for each SOH to meet the desired count
    for target_soh in unique_soh_values:
        soh_cycles = {cycle_num: data for cycle_num, data in all_charge_data[battery_id].items() if data[0, -1] == target_soh}
        current_count = len(soh_cycles)
        print(f'Current count of SOH {target_soh}: {current_count}')

        if current_count < desired_count:
            duplicates_needed = desired_count - current_count
            if duplicates_needed > 0 and current_count > 0:
                chosen_cycles = random.choices(list(soh_cycles.items()), k=duplicates_needed)
                for cycle_num, cycle_data in chosen_cycles:
                    new_cycle_number = max(map(int, all_charge_data[battery_id].keys())) + 1
                    all_charge_data[battery_id][str(new_cycle_number)] = np.copy(cycle_data)

        elif current_count > desired_count:
            excess = current_count - desired_count
            retained_cycles = random.sample(list(soh_cycles.keys()), desired_count)
            for cycle_num in list(soh_cycles.keys()):
                if cycle_num not in retained_cycles:
                    del all_charge_data[battery_id][cycle_num]

        updated_count = len([data for data in all_charge_data[battery_id].values() if data[0, -1] == target_soh])
        print(f"Updated cycle count with SOH {target_soh} for {battery_id}: {updated_count}")

# Calculate and display the total number of cycles across all batteries
total_cycles = sum(len(cycles_data) for cycles_data in all_charge_data.values())
print(f"Total number of cycles in all charge data duplication: {total_cycles}")

# Calculate and store the count of cycles for each SOH value per battery
SOH_cycle_counts = {}
for battery_name, cycles_data in all_charge_data.items():
    SOH_counts = {}
    for cycle_number, cycle_data in cycles_data.items():
        if cycle_data.size > 0:
            current_SOH = cycle_data[0, -1]
            SOH_counts[current_SOH] = SOH_counts.get(current_SOH, 0) + 1
    SOH_cycle_counts[battery_name] = SOH_counts

# Plot the number of cycles vs. SOH for each battery
num_batteries = len(SOH_cycle_counts)
fig, axes = plt.subplots(nrows=num_batteries, figsize=(10, 5 * num_batteries))
if num_batteries == 1:
    axes = [axes]

for ax, (battery_name, counts) in zip(axes, SOH_cycle_counts.items()):
    ax.bar(counts.keys(), counts.values(), color='b')
    ax.set_title(f'Number of Cycles vs. State of Health for {battery_name}')
    ax.set_xlabel('State of Health (%)')
    ax.set_ylabel('Number of Cycles')
    ax.set_xticks(range(0, 101, 5))
    ax.grid(True)

plt.tight_layout()
if showplot:
    plt.show()

# Repeat the plot setup with different subplot dimensions
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
if num_batteries == 1:
    axes = [axes]

for (battery_name, counts), ax in zip(SOH_cycle_counts.items(), axes.flat):
    ax.bar(counts.keys(), counts.values(), color='b')
    ax.set_title(f'Number of Cycles vs. State of Health for {battery_name} - Post Normalisation')
    ax.set_xlabel('State of Health (%)')
    ax.set_ylabel('Number of Cycles')
    ax.set_xticks(range(0, 101, 5))
    ax.grid(True)

plt.tight_layout()
if showplot:
    plt.show()

# Interpolation to give 140 datapoints cycle
def interpolate_cycle_data_fixed_voltages(cycle_data, target_length=140):
    if cycle_data.size == 0:
        return np.full((target_length, 2), 0)

    # Extract Charge Capacity and Voltage data
    Q = cycle_data[:, 0]
    Voltage = cycle_data[:, 1]

    # Define fixed range of Voltage between 3.7V and 4V
    fixed_min_voltage, fixed_max_voltage = 3.7, 4
    target_voltages = np.linspace(fixed_min_voltage, fixed_max_voltage, num=target_length)

    # Interpolation function for original Voltage and Charge Capacity data
    interp_func_IC = interp1d(Voltage, Q, kind='linear', bounds_error=False, fill_value="extrapolate")

    # Interpolate Charge Capacity for target evenly spaced Voltage values
    interpolated_IC = interp_func_IC(target_voltages)

    # Combine interpolated Charge Capacity and new Voltage into a single array
    interpolated_cycle_data = np.column_stack((interpolated_IC, target_voltages))

    return interpolated_cycle_data

# Apply updated interpolation function to each cycle's data for all batteries
for battery_name, cycles_data in all_charge_data.items():
    for cycle_number, cycle_data in cycles_data.items():
        # Ensure data is in NumPy array format
        if not isinstance(cycle_data, np.ndarray):
            cycle_data = cycle_data.to_numpy()

        # Interpolate cycle data to fixed Voltage range
        interpolated_cycle_data = interpolate_cycle_data_fixed_voltages(cycle_data)
        
        # Update the cycle data with the interpolated data
        all_charge_data[battery_name][cycle_number] = interpolated_cycle_data


# Renumber cycles function
def renumber_cycles(all_charge_data):
    renumbered_data = {}
    
    for battery_name, cycles_data in all_charge_data.items():
        renumbered_cycles = {new_index: data for new_index, (_, data) in enumerate(sorted(cycles_data.items(), key=lambda item: int(item[0])))}
        
        renumbered_data[battery_name] = renumbered_cycles
    
    return renumbered_data

all_charge_data = renumber_cycles(all_charge_data)




if export == 1:

    def export_filtered_data_as_text(all_charge_data, base_directory):
        # Create the base directory if it doesn't exist
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)
        
        for battery_name, cycles_data in all_charge_data.items():
            # Generate a sorted list of cycles for processing
            sorted_cycles = sorted(cycles_data.keys())
            
            # Here, cycles_to_export is set to include all cycles; adjust if necessary
            cycles_to_export = sorted_cycles

            # Construct the file path for output
            file_path = os.path.join(base_directory, f"{battery_name}.txt")
            
            with open(file_path, 'w') as file:
                for cycle_number in cycles_to_export:
                    cycle_data = cycles_data[cycle_number]
                    
                    # Extract charge capacity (IC values) and convert to comma-separated string
                    IC_values = cycle_data[:, 0]
                    IC_values_str = ','.join(map(str, IC_values))
                    
                    # Write the IC values to the file with a newline after each cycle
                    file.write(IC_values_str + '\n')
            
            print(f"Exported filtered Q data for {battery_name} to {file_path}")

    # Specify the base directory for exporting data
    export_base_dir = ''
    export_filtered_data_as_text(all_charge_data, export_base_dir)

    def export_battery_cycles_to_csv(all_charge_data, export_directory):
        # Ensure the export directory exists
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)
        
        for battery_name, cycles_data in all_charge_data.items():
            # List to store combined data from all cycles
            combined_data = []
            
            for cycle_number, cycle_data in cycles_data.items():
                # Check to ensure there is data to process
                if cycle_data.size > 0:
                    # Add the cycle number as a new column to the cycle data
                    cycle_data_with_cycle_number = np.column_stack((cycle_data, np.full(cycle_data.shape[0], cycle_number)))
                    combined_data.append(cycle_data_with_cycle_number)

            # Proceed if there is data to export
            if combined_data:
                # Combine all cycle data into a single array
                combined_data_array = np.vstack(combined_data)
                # Convert to DataFrame
                df = pd.DataFrame(combined_data_array, columns=['Charge_Capacity(Ah)', 'Voltage(V)', 'Cycle_Number'])
                # Export to CSV
                file_path = os.path.join(export_directory, f"{battery_name}.csv")
                df.to_csv(file_path, index=False)
                print(f"Exported {battery_name} cycle data to {file_path}")
            else:
                print(f"No data available to export for {battery_name}.")

    # Specify the directory for exporting CSV files
    export_directory = ''
    export_battery_cycles_to_csv(all_charge_data, export_directory)
