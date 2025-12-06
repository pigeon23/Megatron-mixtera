import json
import os
import matplotlib.pyplot as plt

# --- Configuration ---
# Define the base directory where the checkpoint folders are located
SETTING = "1B-default"
BASE_DIR = f"experiments/{SETTING}/ckpt-converted"

# Define the range and step for iteration folders
# Iterations go from 2500 to 10000, step 2500 (i.e., 2500, 5000, 7500, 10000)
START_ITER = 2500
END_ITER = 30000
STEP = 2500

# Define the metrics you want to extract and plot.
# *** IMPORTANT: Adjust these keys based on what is actually in your results.json files. ***
METRICS_TO_PLOT = [
    'arc_easy',
    'hellaswag',
    'openbookqa'
]
# ---------------------

def find_results_file_path(base_dir, depth=3):
    """
    Dynamically finds the full path to the *single* result file by traversing a 
    fixed number of sub-directories (depth), assuming exactly one item at each level.
    The final file name is non-deterministic, but it is the only file/folder in the final directory.
    
    Returns the full path string or None if not found or if the structure is wrong.
    """
    current_path = base_dir
    
    # 1. Traverse the fixed directory structure (L1, L2, L3)
    for level in range(1, depth + 1):
        try:
            # Get the contents of the current directory
            contents = os.listdir(current_path)
        except OSError:
            # Error reading directory (e.g., permission denied)
            return None
            
        # Check the guarantee: exactly one item inside
        if len(contents) != 1:
            print(f"  [WARNING] Level {level} structure error: Expected exactly one item in '{current_path}', found {len(contents)}. Aborting search.")
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
        print(f"  [WARNING] Final directory structure error: Expected exactly one file/folder in '{current_path}', found {len(final_contents)}. Aborting search.")
        return None
    
    # The name of the single file is the only item in final_contents
    final_file_name = final_contents[0]
    file_path = os.path.join(current_path, final_file_name)

    # Check if the path points to a file and return
    if os.path.isfile(file_path):
        # We assume this is the required JSON file, even if the name varies
        return file_path
    else:
        print(f"  [ERROR] Final path '{file_path}' is not a file.")
        return None


def load_and_plot_evaluation_results():
    """
    Walks through the specified iteration folders, finds the single result file
    3 levels deep, extracts defined metrics, and plots them against the iteration number.
    """
    all_data = {metric: [] for metric in METRICS_TO_PLOT}
    iterations = []
    
    # Inform the user about the expected structure based on their clarification
    print(f"Searching for results in: {BASE_DIR}/iter_000****/huggingface_eval/fewshot_0/results/<L1>/<L2>/<L3>/<filename.json>")

    # Generate the list of iteration numbers
    for iteration in range(START_ITER, END_ITER + STEP, STEP):
        # Format the folder name, e.g., 'iter_0002500'
        iter_folder = f"iter_{iteration:07d}"
        
        # Construct the path to the 'eval' directory (the root for dynamic search)
        eval_dir = os.path.join(BASE_DIR, iter_folder, "huggingface_eval/fewshot_0/results")
        
        print(f"Checking evaluation directory: {eval_dir}")

        if not os.path.exists(eval_dir):
            print(f"  [ERROR] Evaluation directory not found: {eval_dir}. Skipping.")
            continue
            
        # Find the full path using the new helper function
        # Note: We no longer pass a filename, as it's discovered dynamically
        file_path = find_results_file_path(eval_dir, depth=1)
        
        if not file_path:
            print(f"  [ERROR] Could not find results file for iteration {iteration}. Skipping.")
            continue
            
        print(f"Attempting to read file: {file_path}")
            
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)["results"]
                
                # Check if all required metrics exist in the JSON
                missing_metrics = [m for m in METRICS_TO_PLOT if m not in results]
                if missing_metrics:
                    print(f"  [WARNING] Results file is missing metrics: {missing_metrics}. Skipping.")
                    continue
                    
                # Store the data
                iterations.append(iteration)
                for metric in METRICS_TO_PLOT:
                    all_data[metric].append(results[metric]["acc,none"])
                
                print(f"  [SUCCESS] Data loaded for iteration {iteration}.")
                
        except json.JSONDecodeError:
            print(f"  [ERROR] Failed to decode JSON from {file_path}. Skipping.")
        except Exception as e:
            print(f"  [ERROR] An unexpected error occurred: {e}. Skipping.")


    if not iterations:
        print("\nNo valid results were found to plot. Ensure the directory structure and JSON keys are correct.")
        return

    # --- Plotting the Results ---
    
    # Create the figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Determine colors for plotting
    colors = plt.cm.get_cmap('Dark2', len(METRICS_TO_PLOT))

    # Use a counter to assign different colors/axes to metrics
    line_styles = ['-o', '--s']  # Solid lines for main axis, dashed for secondary if used
    
    # Plot all metrics on the primary y-axis (ax1) for simplicity
    print("\nStarting plot generation...")
    
    for i, metric in enumerate(METRICS_TO_PLOT):
        ax1.plot(iterations, all_data[metric], line_styles[i % len(line_styles)], color=colors(i), 
                 label=f'{metric.replace("_", " ").title()}')

    # Set up titles and labels
    ax1.set_title(f'Model Performance Over Training Iterations ({SETTING})', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Training Iteration', fontsize=12)
    # Use a generic label for the primary axis since metrics might be mixed
    ax1.set_ylabel('acc,none', fontsize=12) 

    # Add grid, legend, and ticks
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='best', fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    
    # Customize the x-axis ticks to show only the iteration numbers
    ax1.set_xticks(iterations)
    
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig(f'plots/{SETTING}.png')
    print("Plot display finished.")


if __name__ == "__main__":
    
    load_and_plot_evaluation_results()