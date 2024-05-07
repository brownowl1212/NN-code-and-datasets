# Author: Nirosan Pragash
# Master's Dissertation Project
# University: University of Bath
# Course: MEng Integrated Mechanical and Electrical Engineering
# This script is designed to process and analyse battery cycling data for different the CAS dataset.
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


# Base directory where the CSV files are located
base_dir = ''

# CSV file names for each Cell series
csv_files = {
    'Cell1': 'Cell1_allQVs.csv',
    'Cell2': 'Cell2_allQVs.csv',
    'Cell3': 'Cell3_allQVs.csv',
    'Cell4': 'Cell4_allQVs.csv',
    'Cell5': 'Cell5_allQVs.csv',
    'Cell6': 'Cell6_allQVs.csv',
    'Cell7': 'Cell7_allQVs.csv',
    'Cell8': 'Cell8_allQVs.csv'
}

# Load each CSV file into a DataFrame
cells = {name: pd.read_csv(f"{base_dir}/{path}") for name, path in csv_files.items()}

# Dictionary to store voltage and Q data for each cycle and battery
voltage_q_data = {}

# Iterate over each cell
for cell_name, cell_data in cells.items():
    # Get unique cycle indices for the current cell
    cycle_indices = cell_data['Cycle'].unique()
    
    # Initialise dictionary to store data for the current cell
    cell_cycle_data = {}
    
    # Iterate over each cycle index
    for cycle_index in cycle_indices:
        # Extract data for the current cycle index
        cycle_data = cell_data[cell_data['Cycle'] == cycle_index]
        
        # Extract voltage and Q data for the current cycle, dividing Q by 1000
        voltage = cycle_data['V'].to_numpy()
        charge_capacity = cycle_data['Q'].to_numpy() / 1000  # Divide by 1000 to convert to ampere-hours
        
        # Store voltage and Q data for the current cycle
        cell_cycle_data[cycle_index] = {'Voltage(V)': voltage, 'Charge_Capacity(Ah)': charge_capacity}
    
    # Store data for the current cell in the main dictionary
    voltage_q_data[cell_name] = cell_cycle_data

def plot_all_cycles(cell_name):
    if cell_name in voltage_q_data:
        # Group data by Cycle for the specified cell
        grouped_cycles = cells[cell_name].groupby('Cycle')

        # Plot all cycles for the specified cell on a single plot
        plt.figure(figsize=(10, 6))
        for name, group in grouped_cycles:
            plt.plot(group['V'], group['Q'], label=f'Cycle {name}')

        plt.xlabel('Voltage (V)')
        plt.ylabel('Charge Capacity (mAh)')
        plt.title(f'Voltage vs Capacity for all Cycles of {cell_name}')
        plt.grid(True)
        if showplot ==1:
            plt.show()
    else:
        print(f"No data found for {cell_name}.")

# Plot all cycles for each cell individually
for cell_name in voltage_q_data.keys():
    plot_all_cycles(cell_name)

# Iterate through each battery
for battery_name, cycle_data_dict in voltage_q_data.items():
    # Iterate through each cycle of the current battery
    for cycle_number, cycle_data in cycle_data_dict.items():
        # Extract voltage data for the current cycle
        voltage_data = cycle_data['Voltage(V)']

        # Find indices where voltage is within the desired range
        valid_indices = np.where((voltage_data >= 2.7) & (voltage_data <= 4.2))[0]

        # Remove rows where voltage is outside the desired range
        cycle_data['Voltage(V)'] = voltage_data[valid_indices]
        cycle_data['Charge_Capacity(Ah)'] = cycle_data['Charge_Capacity(Ah)'][valid_indices]

        # Update the cycle data for the current battery
        voltage_q_data[battery_name][cycle_number] = cycle_data

# Initialise an empty dictionary to store the maximum Q values for each cycle for each battery
max_q_values = {}

# Iterate through each battery
for battery_name, cycle_data_dict in voltage_q_data.items():
    # Initialise lists to store the maximum Q values and corresponding cycle numbers
    max_q_values[battery_name] = {'Cycle': [], 'Max Q': []}

    # Iterate through each cycle of the current battery
    for cycle_number, cycle_data in cycle_data_dict.items():
        # Extract Q data for the current cycle
        q_data = cycle_data['Charge_Capacity(Ah)']

        # Find the maximum Q value for the current cycle
        max_q_value = max(q_data)

        # Append the cycle number and maximum Q value to the lists
        max_q_values[battery_name]['Cycle'].append(cycle_number)
        max_q_values[battery_name]['Max Q'].append(max_q_value)

# Initialise a figure with subplots
fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)

# Iterate through each battery and its corresponding data
for i, (battery_name, data) in enumerate(max_q_values.items()):
    # Determine subplot index
    row = i // 4  # Calculate row index
    col = i % 4   # Calculate column index

    # Plot the maximum Q values from each cycle for the current battery on its respective subplot
    axs[row, col].plot(data['Cycle'], data['Max Q'], marker='o', linestyle='-')
    axs[row, col].set_xlabel('Cycle Number')
    axs[row, col].set_ylabel('Max Q Value (Ah)')
    axs[row, col].set_title(f'Max Q Values for {battery_name}')
    axs[row, col].grid(True)

# Adjust layout and display the plots
plt.tight_layout()
if showplot == 1:
    plt.show()

soh_values = {}

# Iterate through each battery
for battery_name, cycle_data_dict in voltage_q_data.items():
    # Initialise a list to store the SOH values and corresponding cycle numbers
    soh_values[battery_name] = {'Cycle': [], 'SOH': []}
    first_cycle_max_q = None

    # Iterate through each cycle of the current battery
    for cycle_number, cycle_data in cycle_data_dict.items():
        # Extract Q data for the current cycle
        q_data = cycle_data['Charge_Capacity(Ah)']

        # Find the maximum Q value for the current cycle
        max_q_value = max(q_data)

        # Set the first cycle's max Q as the reference max Q if not already set
        if first_cycle_max_q is None:
            first_cycle_max_q = max_q_value

        # Calculate SOH as the percentage of the current max Q relative to the first cycle's max Q
        soh = (max_q_value / first_cycle_max_q) * 100

        # Append the cycle number and SOH value to the lists
        soh_values[battery_name]['Cycle'].append(cycle_number)
        soh_values[battery_name]['SOH'].append(soh)

# Initialise a figure with subplots
fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)

# Iterate through each battery and its corresponding data
for i, (battery_name, data) in enumerate(soh_values.items()):
    # Determine subplot index
    row = i // 4  # Calculate row index
    col = i % 4   # Calculate column index

    # Plot the SOH values from each cycle for the current battery on its respective subplot
    axs[row, col].plot(data['Cycle'], data['SOH'], marker='o', linestyle='-')
    axs[row, col].set_xlabel('Cycle Number')
    axs[row, col].set_ylabel('SOH (%)')
    axs[row, col].set_title(f'SOH for {battery_name}')
    axs[row, col].grid(True)

# Adjust layout and display the plots
plt.tight_layout()
if showplot == 1:
    plt.show()

def delete_low_soh_cycles(voltage_q_data, threshold_soh=74.9):
    for battery_name, battery_cycles in voltage_q_data.items():
        for cycle_number, cycle_data in list(battery_cycles.items()):
            if 'SOH' in cycle_data and cycle_data['SOH'] < threshold_soh:
                del voltage_q_data[battery_name][cycle_number]
                print(f"Cycle {cycle_number} deleted for {battery_name} due to low SOH.")

# Example usage
delete_low_soh_cycles(voltage_q_data)

def plot_soh_distribution_for_battery(battery_name, voltage_q_data, ax):
    if battery_name in voltage_q_data:
        soh_distribution = {}
        for cycle_number, cycle_data in voltage_q_data[battery_name].items():
            soh = cycle_data['SOH']
            soh_distribution[soh] = soh_distribution.get(soh, 0) + 1
        soh_values = list(soh_distribution.keys())
        cycle_counts = list(soh_distribution.values())
        ax.bar(soh_values, cycle_counts)
        ax.set_xlabel('SOH (%)')
        ax.set_ylabel('Number of Cycles')
        ax.set_title(f"Number of Cycles vs. State of Health for {battery_name}")
        ax.grid(True)
    else:
        print(f"{battery_name} data not found.")

# Create subplots for each battery
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
battery_names = ['Cell1', 'Cell3', 'Cell7', 'Cell8']
for i, ax in enumerate(axes.flat):
    if i < len(battery_names):
        plot_soh_distribution_for_battery(battery_names[i], voltage_q_data, ax)
plt.tight_layout()
if showplot == 1:
    plt.show()

def manage_cycles(battery_data, battery_name):
    battery_cycles = battery_data.get(battery_name, {})
    soh_values_to_process = set(data['SOH'] for data in battery_cycles.values() if 'SOH' in data)
    for soh_value in soh_values_to_process:
        cycles_with_current_soh = {cycle_number: data for cycle_number, data in battery_cycles.items() if 'SOH' in data and data['SOH'] == soh_value}
        existing_cycles_count = len(cycles_with_current_soh)
        cycles_needed = desired_count - existing_cycles_count
        if cycles_needed > 0:
            cycles_to_duplicate = random.choices(list(cycles_with_current_soh.keys()), k=cycles_needed)
            for cycle_number in cycles_to_duplicate:
                max_existing_cycle = max(battery_cycles.keys()) + 1
                battery_cycles[max_existing_cycle] = battery_cycles[cycle_number].copy()
        elif cycles_needed < 0:
            cycles_to_retain = random.sample(list(cycles_with_current_soh.keys()), desired_count)
            cycles_to_remove = set(cycles_with_current_soh.keys()) - set(cycles_to_retain)
            for cycle_number in cycles_to_remove:
                del battery_cycles[cycle_number]
    battery_data[battery_name] = battery_cycles
    for soh_value in soh_values_to_process:
        new_count_soh = sum(1 for data in battery_cycles.values() if 'SOH' in data and data['SOH'] == soh_value)
        print(f"Now there are {new_count_soh} cycles with an SOH of {soh_value} in {battery_name}.")

desired_count = 113
manage_cycles(voltage_q_data, 'Cell1')
manage_cycles(voltage_q_data, 'Cell3')
manage_cycles(voltage_q_data, 'Cell7')
manage_cycles(voltage_q_data, 'Cell8')

def plot_soh_distribution_for_battery(battery_name, voltage_q_data, ax):
    # Check if the battery exists in the voltage_q_data dictionary
    if battery_name in voltage_q_data:
        # Initialise a dictionary to store SOH values and their corresponding cycle counts
        soh_distribution = {}

        # Iterate through each cycle of the battery
        for cycle_number, cycle_data in voltage_q_data[battery_name].items():
            # Extract SOH for the current cycle
            soh = cycle_data['SOH']

            # Increment the count for the SOH value
            soh_distribution[soh] = soh_distribution.get(soh, 0) + 1

        # Extract SOH values and their corresponding cycle counts
        soh_values = list(soh_distribution.keys())
        cycle_counts = list(soh_distribution.values())

        # Plot SOH distribution
        ax.bar(soh_values, cycle_counts)
        ax.set_xlabel('SOH (%)')
        ax.set_ylabel('Number of Cycles')
        ax.set_title(f"Number of Cycles vs. State of Health for {battery_name}")
        ax.grid(True)

    else:
        print(f"{battery_name} data not found.")

# Create subplots for each battery
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot SOH distribution for cells 1, 3, 7, and 8
battery_names = ['Cell1', 'Cell3', 'Cell7', 'Cell8']
for i, ax in enumerate(axes.flat):
    if i < len(battery_names):
        plot_soh_distribution_for_battery(battery_names[i], voltage_q_data, ax)

# Adjust layout
plt.tight_layout()

# Show plots
if showplot == 1:
    plt.show()


def print_soh_cycle_table(battery_name, voltage_q_data):
    # Check if the battery exists in the voltage_q_data dictionary
    if battery_name in voltage_q_data:
        # Initialise a dictionary to store SOH values and their corresponding cycle counts
        soh_distribution = {}

        # Iterate through each cycle of the battery
        for cycle_number, cycle_data in voltage_q_data[battery_name].items():
            # Extract SOH for the current cycle
            soh = cycle_data['SOH']

            # Increment the count for the SOH value
            soh_distribution[soh] = soh_distribution.get(soh, 0) + 1

        # Print table header
        print(f"{'Cycle Number':<15} {'SOH (%)':<15}")

        # Print SOH values and their corresponding cycle numbers
        for soh, cycle_count in soh_distribution.items():
            print(f"{cycle_count:<15} {soh:<15}")

    else:
        print(f"{battery_name} data not found.")

# Print SOH and cycle number table for a specific battery
battery_name_to_print = 'Cell1'  # Change this to the desired battery name
print_soh_cycle_table(battery_name_to_print, voltage_q_data)


def alter_voltage_range(voltage_q_data, original_min=2.75, original_max=4.19, desired_min=3.6, desired_max=4.1):
    adjusted_data = {}
    scale_factor = (desired_max - desired_min) / (original_max - original_min)
    for cell, cycles in voltage_q_data.items():
        adjusted_cycles = {}
        for cycle, data in cycles.items():
            voltage_data = data['Voltage(V)']
            adjusted_voltages = (voltage_data - original_min) * scale_factor + desired_min
            # Ensure the adjusted voltages are clipped to not exceed the desired range
            adjusted_voltages = np.clip(adjusted_voltages, desired_min, desired_max)
            # Update the voltage data with the adjusted values
            voltage_q_data[cell][cycle]['Voltage(V)'] = adjusted_voltages
        adjusted_data[cell] = adjusted_cycles
    return voltage_q_data

voltage_q_data = alter_voltage_range(voltage_q_data)

def interpolate_cycle_data_fixed_voltages(cycle_data, target_length=140, min_charge_capacity=0.001):
    if cycle_data.empty:
        return pd.DataFrame(columns=['Charge_Capacity(Ah)', 'Voltage(V)'], 
                            data=np.full((target_length, 2), np.nan))

    # Dropping duplicates with averaging if needed
    cycle_data = cycle_data.groupby('Voltage(V)', as_index=False).mean()

    # Extract Charge_Capacity(Ah) and Voltage(V)
    Q = cycle_data['Charge_Capacity(Ah)'].values
    V = cycle_data['Voltage(V)'].values

    # Define the target voltages
    fixed_min_voltage, fixed_max_voltage = 3.6, 4.1
    target_voltages = np.linspace(fixed_min_voltage, fixed_max_voltage, num=target_length)

    # Check for sufficient range in V to avoid divide by zero in interpolation
    if np.ptp(V) == 0:
        return pd.DataFrame(columns=['Charge_Capacity(Ah)', 'Voltage(V)'],
                            data=np.column_stack((np.full(target_length, min_charge_capacity), target_voltages)))

    # Interpolation
    interp_func = interp1d(V, Q, kind='linear', bounds_error=False, fill_value="extrapolate")
    interpolated_Q = interp_func(target_voltages)

    # Apply minimum threshold for charge capacity
    interpolated_Q[interpolated_Q < min_charge_capacity] = min_charge_capacity

    # Constructing the DataFrame
    interpolated_data = pd.DataFrame({
        'Charge_Capacity(Ah)': interpolated_Q,
        'Voltage(V)': target_voltages
    })

    return interpolated_data

for cell_name, cycles in voltage_q_data.items():
    for cycle_number, cycle_data in cycles.items():
        cycle_data_df = pd.DataFrame(cycle_data)
        interpolated_cycle_data = interpolate_cycle_data_fixed_voltages(cycle_data_df)
        voltage_q_data[cell_name][cycle_number] = interpolated_cycle_data.to_dict('list')

if export == 1:
    def export_Q_data_as_text(interpolated_all_cells_data, export_base_dir):
        if not os.path.exists(export_base_dir):
            os.makedirs(export_base_dir)  # Create the base directory if it doesn't exist

        for cell_name, cycles_data in interpolated_all_cells_data.items():
            file_path = os.path.join(export_base_dir, f"{cell_name}.txt")
            with open(file_path, 'w') as file:
                for cycle_data in cycles_data.values():
                    # Extract only the Q values from the cycle data
                    Q_values = cycle_data['Charge_Capacity(Ah)']

                    # Convert Q values to a comma-separated string
                    Q_values_str = ','.join(map(str, Q_values))

                    # Write Q values to the file, with a newline character after all values have been written
                    file.write(Q_values_str + '\n')

    export_directory = ''
    export_Q_data_as_text(voltage_q_data, export_directory)


    def export_battery_cycles_to_csv(all_charge_data, export_directory):
        # Ensure the export directory exists
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)

        # Iterate through each battery's data
        for cell_name, cycles_data in all_charge_data.items():
            # Initialise a list to store the combined data for all cycles
            combined_data = []

            # Iterate through each cycle
            for cycle_number, cycle_data in cycles_data.items():
                # Ensure cycle_data has Charge_Capacity(Ah) and Voltage(V) with data
                if cycle_data['Charge_Capacity(Ah)'] and cycle_data['Voltage(V)']:
                    # Convert lists to arrays for stack operation
                    charge_capacity = np.array(cycle_data['Charge_Capacity(Ah)'])
                    voltage = np.array(cycle_data['Voltage(V)'])

                    # Add the cycle number as a new column to the cycle data
                    cycle_data_with_cycle_number = np.column_stack((charge_capacity, voltage, np.full(charge_capacity.shape[0], cycle_number)))

                    # Append this cycle's data to the combined data list
                    combined_data.append(cycle_data_with_cycle_number)

            # If we have collected data, convert it to a DataFrame
            if combined_data:
                # Combine all cycles' data into one array
                combined_data_array = np.vstack(combined_data)

                # Convert the array to a DataFrame
                df = pd.DataFrame(combined_data_array, columns=['Charge_Capacity(Ah)', 'Voltage(V)', 'Cycle_Number'])

                # Export the DataFrame to a CSV file
                file_path = os.path.join(export_directory, f"{cell_name}.csv")
                df.to_csv(file_path, index=False)

    export_directory = ''
    export_battery_cycles_to_csv(voltage_q_data, export_directory)
