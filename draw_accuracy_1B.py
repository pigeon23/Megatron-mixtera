import json
import os
import matplotlib.pyplot as plt

# --- Configuration ---
# Define the list of base setting directories you want to plot
SETTINGS_TO_PLOT = [
    "1B-default-ADO",  # Replace with actual setting names
    "1B-default",
] 
SETTINGS_TO_COLOR = {
    "1B-default-ADO": "royalblue",
    "1B-default": "orange",
}
BASE_PATH_TEMPLATE = "experiments/{setting}/ckpt-converted"

# Define the range and step for iteration folders (same for all settings)
START_ITER = 2500
END_ITER = 30000
STEP = 2500

# Define the metrics you want to extract and plot.
METRICS_TO_PLOT = [
    'hellaswag',
    'openbookqa',
    'arc_easy',
]

# ---------------------

def find_results_file_path(base_dir, depth=1):
    """
    Dynamically finds the full path to the *single* result file by traversing a 
    fixed number of sub-directories (depth), assuming exactly one item at each level.
    """
    current_path = base_dir
    
    # 1. Traverse the fixed directory structure (L1, L2, L3...)
    for level in range(1, depth + 1):
        try:
            contents = os.listdir(current_path)
        except OSError:
            return None
            
        if len(contents) != 1:
            print(f"[WARNING] Level {level} structure error: Expected exactly one item in '{current_path}', found {len(contents)}. Aborting search.")
            return None
            
        # Move down to the next directory
        current_path = os.path.join(current_path, contents[0])

    # 2. Handle the final directory where the result file resides
    try:
        final_contents = os.listdir(current_path)
    except OSError:
        return None

    # Check the guarantee: exactly one item inside the final directory
    if len(final_contents) != 1:
        print(f"[WARNING] Final directory structure error: Expected exactly one file/folder in '{current_path}', found {len(final_contents)}. Aborting search.")
        return None
    
    # The name of the single file is the only item in final_contents
    final_file_name = final_contents[0]
    file_path = os.path.join(current_path, final_file_name)

    # Check if the path points to a file and return
    if os.path.isfile(file_path):
        return file_path
    else:
        print(f"[ERROR] Final path '{file_path}' is not a file.")
        return None


def load_setting_results(setting, iterations_list):
    """
    Walks through the specified iteration folders for a single setting, 
    extracts defined metrics, and returns the data.
    """
    BASE_DIR = BASE_PATH_TEMPLATE.format(setting=setting)
    # Data will be stored as: {metric_name: [list_of_values]}
    setting_data = {metric: [] for metric in METRICS_TO_PLOT}
    
    print(f"\n--- Loading results for SETTING: {setting} ---")

    successful_iterations = []

    for iteration in iterations_list:
        # Format the folder name, e.g., 'iter_0002500'
        iter_folder = f"iter_{iteration:07d}"
        
        # Construct the path to the 'eval' directory (the root for dynamic search)
        eval_dir = os.path.join(BASE_DIR, iter_folder, "huggingface_eval/fewshot_0/results")
        
        if not os.path.exists(eval_dir):
            continue
            
        # Find the full path (depth=1 for L1/<filename.json>)
        file_path = find_results_file_path(eval_dir, depth=1)
        
        if not file_path:
            continue
            
        try:
            with open(file_path, 'r') as f:
                # Assuming the structure is: {"results": {"metric_name": {"acc,none": value, ...}, ...}}
                results = json.load(f)["results"]
                
                # Check if all required metrics exist in the JSON
                missing_metrics = [m for m in METRICS_TO_PLOT if m not in results]
                if missing_metrics:
                    continue
                    
                # Store the data
                successful_iterations.append(iteration)
                for metric in METRICS_TO_PLOT:
                    # Append the accuracy value for the metric
                    # NOTE: This assumes 'acc,none' is the correct key for accuracy
                    metric_key = 'acc,none'
                    if metric in results and metric_key in results[metric]:
                         setting_data[metric].append(results[metric][metric_key])
                    else:
                         # Skip this iteration if the expected key is missing for this metric
                         # We must revert the successful_iteration append if we skip data
                         successful_iterations.pop() 
                         break 
                
        except json.JSONDecodeError:
            print(f"[ERROR] Failed to decode JSON from {file_path} for iteration {iteration}. Skipping.")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred for iteration {iteration}: {e}. Skipping.")

    # Trim all metric data lists to match the number of successful iterations
    # (This ensures all lists remain synchronized)
    final_data = {}
    for metric, data_list in setting_data.items():
        final_data[metric] = data_list[:len(successful_iterations)] 
        
    return successful_iterations, final_data


def plot_all_results(all_results):
    """
    Plots the collected results for all settings, with each metric in a separate subplot.
    all_results structure: {setting_name: {iterations: [list], metrics: {metric_name: [list_of_values]}}}
    
    MODIFICATION: Adds the legend ONLY to the lower right of the LAST subplot.
    """
    
    if not any(all_results.values()):
        print("\nNo valid results were found to plot. Ensure the directory structure and JSON keys are correct.")
        return

    # --- Setup Subplots ---
    n_metrics = len(METRICS_TO_PLOT)
    
    # Use standard figsize
    fig, axes = plt.subplots(1, n_metrics, figsize=(7.5 * n_metrics, 5.5))
    
    # Ensure 'axes' is always an iterable array, even if n_metrics is 1
    if n_metrics == 1:
        axes = [axes]
    
    # Define a color map for settings
    setting_colors = plt.cm.get_cmap('Dark2', len(SETTINGS_TO_PLOT))
    line_style = '-o' 

    print("\nStarting plot generation (each metric in a subplot)...")

    # Get unique, sorted iteration values for X-axis ticks (across all successful runs)
    all_iterations = []
    for setting in all_results:
        all_iterations.extend(all_results[setting]['iterations'])
    x_ticks = sorted(list(set(all_iterations)))
    
    # Variables to store legend information (not needed if using ax.legend for the last plot)
    # handles = []
    # labels = []


    # 1. Iterate over each METRIC to create a subplot
    for metric_idx, metric in enumerate(METRICS_TO_PLOT):
        ax = axes[metric_idx]
        
        line_counter = 0

        # 2. Iterate over each SETTING and plot its line on the current metric's subplot
        for setting_idx, setting in enumerate(SETTINGS_TO_PLOT):
            setting_data = all_results.get(setting)
            
            if not setting_data or not setting_data['iterations']:
                continue
                
            # Check if data exists for the current metric in this setting
            if metric not in setting_data['metrics'] or not setting_data['metrics'][metric]:
                 continue

            iterations = setting_data['iterations']
            y_values = setting_data['metrics'][metric]
            
            # Define the color for the current setting
            color_i = SETTINGS_TO_COLOR.get(setting, setting_colors(setting_idx))
            
            # Plot the line/points
            # Use the 'setting' name as the label in ALL subplots, but the legend 
            # will only be explicitly called for the last one.
            line, = ax.plot(iterations, y_values, line_style, color=color_i, 
                             label=f'{setting}', linewidth=2)
            
            # The original code collected handles/labels from the LAST subplot (metric_idx == 2)
            # This logic can be removed/simplified since we'll call ax.legend() directly.
            # if metric_idx == 2:
            #     handles.append(line)
            #     labels.append(setting)

            # 3. Add annotation for each point
            for x, y in zip(iterations, y_values):
                label = f"{y:.3f}"
                
                # Annotate the point (x, y) with a small offset
                ax.annotate(label, (x, y), textcoords="offset points", 
                             # Vary the offset slightly for clearer annotations when lines are close
                             xytext=(5, 5 + 5 * (line_counter % 3)), 
                             ha='left', fontsize=7, color='gray') 
            
            line_counter += 1

        # --- Subplot specific configuration ---
        ax.set_title(metric, fontsize=12) # Add title for clarity
        ax.set_ylabel('Accuracy', fontsize=10) 
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # *** ADDED: Place the legend in the lower right of the LAST subplot ***
        if metric_idx == n_metrics - 1:
             ax.legend(loc='lower right', 
                       title="Settings", 
                       fontsize=9, 
                       title_fontsize=10, 
                       frameon=True)


    # --- Final Figure Configuration (applies to the bottom plot) ---
    
    # Set X-axis label only on the rightmost plot (if not sharing x-axis)
    axes[1].set_xlabel('Training Iteration', fontsize=12) 
    
    # Customize the x-axis ticks to show all unique iteration numbers
    # Apply to all axes for consistency, or just the last one
    if x_ticks:
        for ax in axes:
             ax.set_xticks(x_ticks)
    
    # Rotate ticks for better readability, applied to all x-labels
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # --- REMOVED: fig.legend(...) to meet the requirement ---
    # if handles:
    #     fig.legend(handles, labels, 
    #                loc='lower right', 
    #                ncol=len(SETTINGS_TO_PLOT), 
    #                title="Settings", 
    #                fontsize=10, 
    #                title_fontsize=10,
    #                frameon=True)
        
    plt.tight_layout() # Use tight_layout without rect since fig.legend is gone
    
    # Ensure the 'plots' directory exists before saving
    os.makedirs('plots', exist_ok=True)
    
    # Create a generic filename based on metrics
    metrics_label = "_".join(METRICS_TO_PLOT[:3])
    save_filename = f'plots/multi_setting_subplot_eval_{metrics_label}_subplot_legend.png'
    
    plt.savefig(save_filename)
    print(f"\nPlot saved to {save_filename}")
    print("Plot display finished.")


if __name__ == "__main__":
    
    # Pre-generate the full list of expected iterations
    expected_iterations = list(range(START_ITER, END_ITER + STEP, STEP))
    
    # Dictionary to hold all collected data: 
    # {setting: {iterations: [list], metrics: {metric: [list]}}}
    ALL_RESULTS_DATA = {}

    for setting in SETTINGS_TO_PLOT:
        iterations, data = load_setting_results(setting, expected_iterations)
        ALL_RESULTS_DATA[setting] = {
            'iterations': iterations,
            'metrics': data
        }

    plot_all_results(ALL_RESULTS_DATA)