# Author: Nirosan Pragash
# Master's Dissertation Project
# University: University of Bath
# Course: MEng Integrated Mechanical and Electrical Engineering
# This script is designed to process and analyse battery cycling data for different the Oxford dataset.
# It includes data loading, preprocessing, analysis, and visualisation components. Functions
# are provided for plotting cycle data, calculating state of health (SOH), and exporting results.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
import random

# Toggles for exporting data and showing plots
showplot = 0
export = 0

batt07_cccv = pd.read_csv(f"")

# Define the columns to keep
columns_to_keep = ['Cycle_Index', 'Voltage(V)', 'Charge_Capacity(Ah)']
batt07_cccv = batt07_cccv[batt07_cccv['Current(A)'] > 0][columns_to_keep]

# List of all CS2_ batteries DataFrames
batteries = [batt07_cccv]
battery_names = ['batt07_cccv']

# Structure to hold each battery's cycles matrices
all_charge_data = {}

for battery, name in zip(batteries, battery_names):
    cycles = battery['Cycle_Index'].unique()
    battery_cycles_matrices = {}
    
    for cycle in cycles:
        cycle_data = battery[battery['Cycle_Index'] == cycle][['Charge_Capacity(Ah)', 'Voltage(V)']]
        battery_cycles_matrices[cycle] = cycle_data.to_numpy()
    all_charge_data[name] = battery_cycles_matrices

# Iterate through each battery and cycle in all_charge_data
for battery, cycles_data in all_charge_data.items():
    for cycle, cycle_data in cycles_data.items():
        Q_data = cycle_data[:, 0]  # First column contains Q
        Voltage_data = cycle_data[:, 1]  # Second column contains Voltage(V)
        
        # Find indices where Q values are negative 
        negative_Q_indices = np.where(Q_data < 0)[0]
        
        # Remove rows where Q values are negative 
        cycle_data = np.delete(cycle_data, negative_Q_indices, axis=0)
        
        # Update the modified cycle data back to all_charge_data
        all_charge_data[battery][cycle] = cycle_data

# Iterate through each battery and print the number of cycles
for battery, cycles_data in all_charge_data.items():
    num_cycles = len(cycles_data)
    print(f"Number of cycles for {battery}: {num_cycles}")

# Initialise a dictionary to store the count of cycles for each battery
full_cycle_count = {}

# Iterate through each battery in all_charge_data to count the cycles
for battery, cycles_data in all_charge_data.items():
    cycle_count = len(cycles_data)  # Count the number of cycles for the current battery
    full_cycle_count[battery] = cycle_count  # Store the cycle count in the dictionary


# Adjust Q values for each cycle
for battery_name, cycles_data in all_charge_data.items():
    for cycle_number, cycle_data in cycles_data.items():
        if cycle_data.size > 0:  # Check if there are any data points in the cycle
            min_q_value = cycle_data[:, 0].min()  # Find the lowest Q value in the cycle
            cycle_data[:, 0] -= min_q_value  # Subtract the lowest Q value from all Q values in the cycle
            all_charge_data[battery_name][cycle_number] = cycle_data  # Update the cycle data with the adjusted Q values

# Plotting maximum capacity (Q value) for each cycle for all batteries
plt.figure(figsize=(20, 15))  # Set the size of the figure
for i, (battery_name, cycles_data) in enumerate(all_charge_data.items(), start=1):
    cycle_numbers = []  # List to store cycle numbers
    max_capacities = []  # List to store maximum capacities for each cycle
    for cycle_number, cycle_data in cycles_data.items():
        if cycle_data.size > 0:
            max_capacity = cycle_data[:, 0].max()  # Find the maximum capacity in the current cycle
            cycle_numbers.append(cycle_number)  # Append cycle number to the list
            max_capacities.append(max_capacity)  # Append max capacity to the list
    plt.subplot(2, 2, i)  # Create a subplot for the current battery
    plt.plot(cycle_numbers, max_capacities, marker='o', linestyle='-', color='b')
    plt.title(f'Maximum Capacity vs. Cycle for {battery_name}')
    plt.xlabel('Cycle Number')
    plt.ylabel('Maximum Capacity (Ah)')
    plt.grid(True)
plt.tight_layout()  # Adjust layout so plots do not overlap
if showplot == 1:
    plt.show()

# Initialise the dictionary to store the first cycle maximum capacity for each battery
SOH100_capacity = {}

# Iterate through all charge data to establish the baseline max capacity from the first cycle
for battery_name, cycles_data in all_charge_data.items():
    first_cycle_data = cycles_data[min(cycles_data, key=int)]  # Find the first cycle data using the minimum cycle number
    if first_cycle_data.size > 0:
        SOH100_capacity[battery_name] = first_cycle_data[:, 0].max()  # Set the maximum capacity of the first cycle for each battery

# Iterate through all charge data again to update each cycle's data with SOH
for battery_name, cycles_data in all_charge_data.items():
    for cycle_number, cycle_data in sorted(cycles_data.items(), key=lambda x: int(x[0])):
        if cycle_data.size > 0:
            max_capacity = cycle_data[:, 0].max()
            if max_capacity > 0 and battery_name in SOH100_capacity:
                SOH = (max_capacity / SOH100_capacity[battery_name]) * 100  # Calculate SOH based on the first cycle's max capacity
                SOH = int(SOH // 2) * 2  # Round SOH to the nearest multiple of 2
                cycle_data_with_SOH = np.column_stack((cycle_data, np.full_like(cycle_data[:, 0], SOH)))  # Update the cycle data with SOH as the third column
                all_charge_data[battery_name][cycle_number] = cycle_data_with_SOH  # Update the cycle data in all_charge_data

# Desired count of cycles for each SOH
desired_count = 210

# Iterate over all batteries in the all_charge_data dictionary
for battery_id in all_charge_data.keys():
    print(f"Processing battery: {battery_id}")
    
    # Identify all unique SOH values for the current battery
    unique_soh_values = set(data[0, -1] for data in all_charge_data[battery_id].values() if data.size > 0)
    
    # Process each SOH value found in the battery's data
    for target_soh in unique_soh_values:
        # Extract cycles with the current SOH for the specified battery
        soh_cycles = {cycle_num: data for cycle_num, data in all_charge_data[battery_id].items() if data[0, -1] == target_soh}

        # Current count of cycles with the current SOH
        current_count = len(soh_cycles)
        print(f'Current count of SOH {target_soh}: {current_count}')

        if current_count < desired_count:
            # Calculate how many duplicates are needed
            duplicates_needed = desired_count - current_count
            # Choose random cycles to duplicate if there are any cycles, otherwise skip
            if duplicates_needed > 0 and current_count > 0:
                chosen_cycles = random.choices(list(soh_cycles.items()), k=duplicates_needed)

                # Duplicate the chosen cycles and add them to the data
                for cycle_num, cycle_data in chosen_cycles:
                    # Find a new unique cycle number
                    new_cycle_number = max(map(int, all_charge_data[battery_id].keys())) + 1
                    # Add the duplicated cycle with a unique cycle number
                    all_charge_data[battery_id][str(new_cycle_number)] = np.copy(cycle_data)

        elif current_count > desired_count:
            # Calculate how many cycles to remove
            excess = current_count - desired_count
            # Randomly choose cycles to retain
            retained_cycles = random.sample(list(soh_cycles.keys()), desired_count)
            
            # Update the soh_cycles to keep only the randomly chosen cycles
            for cycle_num in list(soh_cycles.keys()):
                if cycle_num not in retained_cycles:
                    del all_charge_data[battery_id][cycle_num]

        # Verify the update for the current SOH
        updated_count = len([data for data in all_charge_data[battery_id].values() if data[0, -1] == target_soh])
        print(f"Updated cycle count with SOH {target_soh} for {battery_id}: {updated_count}")

# Dictionary to store the count of cycles for each SOH value per battery
SOH_cycle_counts = {}

# Iterate through all charge data to count cycles for each SOH value
for battery_name, cycles_data in all_charge_data.items():
    SOH_counts = {}  # Temporary dictionary to store SOH counts for the current battery
    for cycle_number, cycle_data in cycles_data.items():
        if cycle_data.size > 0:
            # Retrieve the SOH from the third column
            current_SOH = cycle_data[0, -1]
            if current_SOH in SOH_counts:
                SOH_counts[current_SOH] += 1
            else:
                SOH_counts[current_SOH] = 1
    # Store the counts for each SOH value for the current battery in the main dictionary
    SOH_cycle_counts[battery_name] = SOH_counts

# Determine number of batteries to set up subplot dimensions
num_batteries = len(SOH_cycle_counts)
fig, axes = plt.subplots(nrows=num_batteries, figsize=(10, 5 * num_batteries))

# Check if there's only one battery to avoid indexing error
if num_batteries == 1:
    axes = [axes]

# Plot the data for each battery
for ax, (battery_name, counts) in zip(axes, SOH_cycle_counts.items()):
    ax.bar(counts.keys(), counts.values(), color='b')
    ax.set_title(f'Number of Cycles vs. State of Health for {battery_name}')
    ax.set_xlabel('State of Health (%)')
    ax.set_ylabel('Number of Cycles')
    ax.set_xticks(range(0, 101, 5))  # Set x-axis ticks in increments of 5
    ax.grid(True)

plt.tight_layout()
if showplot == 1:
    plt.show()

def renumber_cycles(all_charge_data):
    renumbered_data = {}
    
    for battery_name, cycles_data in all_charge_data.items():
        # Renumber the cycles starting from 0
        renumbered_cycles = {new_index: data for new_index, (_, data) in enumerate(sorted(cycles_data.items(), key=lambda item: int(item[0])))}
        # Update the dataset for the current battery
        renumbered_data[battery_name] = renumbered_cycles
    
    return renumbered_data

# Apply the function to renumber the cycles
all_charge_data = renumber_cycles(all_charge_data)

def compress_voltage_range(all_charge_data, original_min=3.17, original_max=3.65, desired_min=3.6, desired_max=4.1):
    adjusted_data = {}
    scale_factor = (desired_max - desired_min) / (original_max - original_min)
    for battery_name, cycles_data in all_charge_data.items():
        adjusted_cycle_data = {}
        for cycle, data in cycles_data.items():
            voltage_data = data[:, 1]
            adjusted_voltages = (voltage_data - original_min) * scale_factor + desired_min
            # Ensure the adjusted voltages are clipped to not exceed the desired range
            adjusted_voltages = np.clip(adjusted_voltages, desired_min, desired_max)
            # Update the voltage data with the adjusted values
            all_charge_data[battery_name][cycle][:, 1] = adjusted_voltages
        adjusted_data[battery_name] = adjusted_cycle_data
    return all_charge_data

# Apply the compression to adjust the voltage values for each cycle in each cell
all_charge_data = compress_voltage_range(all_charge_data)

# Interpolation to give 140 datapoints per cycle
def interpolate_cycle_data_fixed_voltages(cycle_data, target_length=140):
    if cycle_data.size == 0:
        return np.full((target_length, 2), 0)

    # Extract Charge_Capacity(Ah) and Voltage data
    Q = cycle_data[:, 0]
    Voltage = cycle_data[:, 1]

    # Fixed range of Voltage between 3.6V and 4.1V
    fixed_min_voltage, fixed_max_voltage = 3.6, 4.1
    target_voltages = np.linspace(fixed_min_voltage, fixed_max_voltage, num=target_length)

    # Create an interpolation function based on the original Voltage and Q data
    interp_func_IC = interp1d(Voltage, Q, kind='linear', bounds_error=False, fill_value="extrapolate")

    # Interpolate Q data for the target, evenly spaced Voltage values
    interpolated_IC = interp_func_IC(target_voltages)

    # Combine interpolated Q and new evenly spaced Voltage into a single array
    interpolated_cycle_data = np.column_stack((interpolated_IC, target_voltages))

    return interpolated_cycle_data

# Apply this updated interpolation function to each cycle's data for all batteries
for battery_name, cycles_data in all_charge_data.items():
    for cycle_number, cycle_data in cycles_data.items():
        # Convert DataFrame to NumPy array if it's not already
        if not isinstance(cycle_data, np.ndarray):
            cycle_data = cycle_data.to_numpy()

        # Interpolate the cycle data to 500 data points with fixed Voltage(V) range
        interpolated_cycle_data = interpolate_cycle_data_fixed_voltages(cycle_data)

        # Update the cycle data with the interpolated data
        all_charge_data[battery_name][cycle_number] = interpolated_cycle_data

# Adjust Q values for each cycle
for battery_name, cycles_data in all_charge_data.items():
    for cycle_number, cycle_data in cycles_data.items():
        # Check if there are any data points in the cycle
        if cycle_data.size > 0:
            # Find the lowest Q value in the cycle
            min_q_value = cycle_data[:, 0].min()
            # Subtract the lowest Q value from all Q values in the cycle
            cycle_data[:, 0] -= min_q_value
            # Update the cycle data with the adjusted Q values
            all_charge_data[battery_name][cycle_number] = cycle_data

if export == 1:
    # Export the dataframes
    def export_filtered_data_as_text(all_charge_data, base_directory):
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)  # Create the base directory if it doesn't exist
        
        for battery_name, cycles_data in all_charge_data.items():
            sorted_cycles = sorted(cycles_data.keys())  # Get a sorted list of cycles
            
            # Define a file name for each battery
            file_path = os.path.join(base_directory, f"{battery_name}.txt")
            
            with open(file_path, 'w') as file:
                for cycle_number in sorted_cycles:
                    cycle_data = cycles_data[cycle_number]
                    
                    # Extract Q values from the cycle data
                    IC_values = cycle_data[:, 0] 
                    
                    # Convert Q values to a comma-separated string
                    IC_values_str = ','.join(map(str, IC_values))
                    
                    # Write Q values to the file, with a newline character for each new cycle
                    file.write(IC_values_str + '\n')
            
            print(f"Exported filtered Q data for {battery_name} to {file_path}")

    export_base_dir = ''
    export_filtered_data_as_text(all_charge_data, export_base_dir)

    # Prepare data for export
    cycles_count_data = {}  # Dictionary to hold count data

    for battery_name, cycles_data in all_charge_data.items():
        cycle_counts = {cycle: data.shape[0] for cycle, data in cycles_data.items()}
        cycles_count_data[battery_name] = pd.DataFrame(list(cycle_counts.items()), columns=['Cycle', 'Number of Values'])

    # Export the counts to an Excel document using a default Excel writer
    with pd.ExcelWriter('battery_cycle_counts.xlsx') as writer:
        for battery_name, count_df in cycles_count_data.items():
            count_df.to_excel(writer, sheet_name=battery_name, index=False)

    def export_battery_cycles_to_csv(all_charge_data, export_directory):
        # Ensure the export directory exists
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)
        
        # Iterate through each battery's data
        for battery_name, cycles_data in all_charge_data.items():
            combined_data = []  # Initialise a list to store the combined data for all cycles
            
            for cycle_number, cycle_data in cycles_data.items():
                if cycle_data.size > 0:  # Ensure cycle_data is not empty
                    # Add the cycle number as a new column to the cycle data
                    cycle_data_with_cycle_number = np.column_stack((cycle_data, np.full(cycle_data.shape[0], cycle_number)))
                    combined_data.append(cycle_data_with_cycle_number)  # Append this cycle's data to the combined data list

            if combined_data:
                combined_data_array = np.vstack(combined_data)  # Combine all cycles' data into one array
                df = pd.DataFrame(combined_data_array, columns=['Charge_Capacity(Ah)', 'Voltage(V)', 'Cycle_Number'])
                file_path = os.path.join(export_directory, f"{battery_name}.csv")
                df.to_csv(file_path, index=False)  # Export the DataFrame to a CSV file
                print(f"Exported {battery_name} cycle data to {file_path}")
            else:
                print(f"No data available to export for {battery_name}.")

    export_directory = ''
    export_battery_cycles_to_csv(all_charge_data, export_directory)

