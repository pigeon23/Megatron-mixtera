import json
import os
import matplotlib.pyplot as plt

# --- Configuration (Unchanged) ---
SETTING1 = "3B-default-ADO"
SETTING2 = "3B-default"
BASE_DIR1 = f"experiments/{SETTING1}/ckpt-converted"
BASE_DIR2 = f"experiments/{SETTING2}/ckpt-converted"

START_ITER = 2500
END_ITER = 30000
STEP = 2500

METRICS_TO_PLOT = [
    'arc_easy',
    'hellaswag',
    'openbookqa'
]
# ---------------------

# find_results_file_path function (Unchanged)
def find_results_file_path(base_dir, depth=1):
    current_path = base_dir
    
    for level in range(1, depth + 1):
        try:
            contents = os.listdir(current_path)
        except OSError:
            return None
            
        if len(contents) != 1:
            print(f"[WARNING] Level {level} structure error: Expected exactly one item in '{current_path}', found {len(contents)}. Aborting search.")
            return None
            
        current_path = os.path.join(current_path, contents[0])

    try:
        final_contents = os.listdir(current_path)
    except OSError:
        return None

    if len(final_contents) != 1:
        print(f"[WARNING] Final directory structure error: Expected exactly one file/folder in '{current_path}', found {len(final_contents)}. Aborting search.")
        return None
    
    final_file_name = final_contents[0]
    file_path = os.path.join(current_path, final_file_name)

    if os.path.isfile(file_path):
        return file_path
    else:
        print(f"[ERROR] Final path '{file_path}' is not a file.")
        return None

# collect_data_for_setting function (Unchanged)
def collect_data_for_setting(base_dir, setting_name, start_iter, end_iter, step, metrics_to_plot):
    all_data = {metric: [] for metric in metrics_to_plot}
    iterations = []
    
    print(f"\n--- Collecting data for setting: {setting_name} ---")

    for iteration in range(start_iter, end_iter + step, step):
        iter_folder = f"iter_{iteration:07d}"
        
        eval_dir = os.path.join(base_dir, iter_folder, "huggingface_eval/fewshot_0/results")
        
        print(f"Checking directory: {eval_dir}")

        if not os.path.exists(eval_dir):
            print(f"[ERROR] Evaluation directory not found: {eval_dir}. Skipping.")
            continue
            
        file_path = find_results_file_path(eval_dir, depth=1)
        
        if not file_path:
            print(f"[ERROR] Could not find results file for iteration {iteration}. Skipping.")
            continue
            
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)["results"]
                
                missing_metrics = [m for m in metrics_to_plot if m not in results]
                if missing_metrics:
                    print(f"[WARNING] Results file is missing metrics: {missing_metrics}. Skipping.")
                    continue
                    
                iterations.append(iteration)
                for metric in metrics_to_plot:
                    all_data[metric].append(results[metric]["acc,none"])
                
                print(f"[SUCCESS] Data loaded for iteration {iteration}.")
                
        except json.JSONDecodeError:
            print(f"[ERROR] Failed to decode JSON from {file_path}. Skipping.")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {e}. Skipping.")
            
    return iterations, all_data

# plot_data_on_axes function (Modified for modularity in legend creation)
def plot_data_on_axes(ax, iterations, all_data, setting_name, colors_cmap, line_styles, offset_x=0):
    """
    Plots the collected data for a single setting onto a given Matplotlib axis. 
    It returns the generated lines (handles) and labels for custom legend creation.
    """
    num_metrics = len(all_data)
    colors = plt.cm.get_cmap(colors_cmap, num_metrics)
    
    lines_and_labels = []
    
    for i, metric in enumerate(all_data.keys()):
        y_values = all_data[metric]
        color_i = colors(i)
        
        # Plot the line/points and capture the line object
        line, = ax.plot(iterations, y_values, line_styles[i % len(line_styles)], color=color_i, 
                        label=f'{metric.replace("_", " ").title()} {setting_name}')
        
        # Store the line object and its corresponding label
        lines_and_labels.append((line, line.get_label()))

        # Add annotation for each point
        for x, y in zip(iterations, y_values):
            x_annot = x + offset_x
            label = f"{y:.3f}"
            
            ax.annotate(label, 
                        (x_annot, y), 
                        textcoords="offset points", 
                        xytext=(5, 5), 
                        ha='left', 
                        fontsize=10, 
                        color=color_i)
                        
    return lines_and_labels


def load_and_plot_evaluation_results():
    """
    Orchestrates the data collection and plotting for both SETTING1 and SETTING2,
    and groups legend items by metric.
    """
    
    # 1. Collect Data
    iterations1, data1 = collect_data_for_setting(
        BASE_DIR1, SETTING1, START_ITER, END_ITER, STEP, METRICS_TO_PLOT
    )
    iterations2, data2 = collect_data_for_setting(
        BASE_DIR2, SETTING2, START_ITER, END_ITER, STEP, METRICS_TO_PLOT
    )

    if not iterations1 and not iterations2:
        print("\nNo valid results were found to plot. Ensure the directory structure and JSON keys are correct.")
        return

    # --- Plotting and Legend Setup ---
    print("\nStarting plot generation...")
    
    line_styles = ['-o', '--s', '-^', ':d', '-P'] 
    
    # Create the figure and axes, making the figure slightly wider to accommodate the external legend
    fig, ax1 = plt.subplots(figsize=(14, 7)) 

    # Plot all data but capture the line handles and labels
    lines_and_labels1 = []
    lines_and_labels2 = []

    # Plot SETTING1 (Primary/ADO)
    if iterations1:
        lines_and_labels1 = plot_data_on_axes(ax1, iterations1, data1, SETTING1, 'Dark2', line_styles, offset_x=0)

    # Plot SETTING2 (Legacy/Default) 
    if iterations2:
        lines_and_labels2 = plot_data_on_axes(ax1, iterations2, data2, SETTING2, 'turbo', line_styles, offset_x=50)

    # --- Legend Reordering Logic ---
    all_lines = []
    all_labels = []

    # Iterate through the metrics in the desired order (from METRICS_TO_PLOT)
    for metric_key in METRICS_TO_PLOT:
        metric_title = metric_key.replace("_", " ").title()
        
        # Find the line/label for SETTING1 for this metric and append it
        label1 = f'{metric_title} {SETTING1}'
        for line, label in lines_and_labels1:
            if label == label1:
                all_lines.append(line)
                all_labels.append(label)
                break
                
        # Find the line/label for SETTING2 for this metric and append it
        label2 = f'{metric_title} {SETTING2}'
        for line, label in lines_and_labels2:
            if label == label2:
                all_lines.append(line)
                all_labels.append(label)
                break
                
    # --- Apply Reordered Legend ---
    # The handles (all_lines) and labels (all_labels) are now in the desired order
    ax1.legend(all_lines, all_labels, 
               loc='upper left', 
               bbox_to_anchor=(1, 0.6), 
               fontsize=15)


    # Set up titles and labels (Unchanged)
    ax1.set_title(f'Model Performance Over Training Iterations ({SETTING1} vs {SETTING2})', fontsize=22, fontweight='bold')
    ax1.set_xlabel('Training Iteration', fontsize=22)
    ax1.set_ylabel('Accuracy (acc,none)', fontsize=22) 

    # Add grid, ticks, and bounds (Unchanged)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    all_iterations = sorted(list(set(iterations1) | set(iterations2)))
    ax1.set_xticks(all_iterations)
    
    x_min = min(all_iterations) - STEP/2
    x_max = max(all_iterations) + STEP/2 + 50 
    ax1.set_xlim(x_min, x_max)
    
    ax1.tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust the figure area for the plot to make space for the legend
    
    # Ensure the 'plots' directory exists before saving
    os.makedirs('plots', exist_ok=True)
    
    plt.savefig(f'plots/{SETTING1}_vs_{SETTING2}.png')
    print(f"\nPlot saved to plots/{SETTING1}_vs_{SETTING2}.png")
    print("Plot display finished.")


if __name__ == "__main__":
    load_and_plot_evaluation_results()