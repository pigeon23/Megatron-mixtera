import re
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

def parse_training_log_block(log_block: str) -> Dict[str, Any]:
    """
    Parses a single log block, ensuring that 'mixtera average feedback time' 
    is associated with the subsequent main iteration metrics.
    
    Args:
        log_block: A string containing a single, complete log entry.
        
    Returns:
        A dictionary containing the parsed metrics for the block.
    """
    parsed_data: Dict[str, Any] = {}
    lines = log_block.strip().split('\n')
    current_time_block = None

    # --- Regex patterns (same as before) ---
    # Simplified to capture the time for better robustness against varying preceding text
    feedback_time_pattern = re.compile(r"mixtera average feedback time\s+(\d+\.\d+)s") 
    wait_time_pattern = re.compile(r"mixtera average wait time\s+(\d+\.\d+)s") 
    main_metrics_pattern = re.compile(
        r"\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\]\s+iteration\s+(\d+)/\s+(\d+)\s+\|\s+"
        r"consumed samples:\s+(\d+)\s+\|\s+elapsed time per iteration \(ms\):\s+(\d+\.\d+)\s+\|\s+"
        r"learning rate:\s+([\d\.E\-\+]+)\s+\|\s+global batch size:\s+(\d+)\s+\|\s+"
        r"lm loss:\s+([\d\.E\-\+]+)\s+\|\s+loss scale:\s+([\d\.E\-\+]+)\s+\|\s+"
        r"grad norm:\s+([\d\.E\-\+]+)\s+\|\s+number of skipped iterations:\s+(\d+)\s+\|\s+"
        r"number of nan iterations:\s+(\d+)"
    )
    time_block_start_pattern = re.compile(r"^\s*([a-z\-]+):\s*$") 
    rank_time_pattern = re.compile(r"^\s*rank\s+(\d+):\s+([\d\.]+)$") 
    
    # --- Parsing Loop ---
    for line in lines:
        line = line.strip()

        # 1. Parse initial feedback time (PRIORITY)
        # We look for this metric first since it comes before the main line.
        if 'mixtera average feedback time' in line:
            match = feedback_time_pattern.search(line)
            if match:
                parsed_data['mixtera'] = float(match.group(1)) * 1000  # Convert to ms
            continue
        
        if 'mixtera average wait time' in line:
            match = wait_time_pattern.search(line)
            if match:
                parsed_data['mixtera_wait'] = float(match.group(1)) * 1000  # Convert to ms
            continue
            
        # 2. Parse main metrics line (Timestamp/Iteration)
        if 'iteration' in line and 'elapsed time' in line:
            match = main_metrics_pattern.search(line)
            if match:
                parsed_data['timestamp'] = match.group(1)
                parsed_data['iteration_current'] = int(match.group(2))
                parsed_data['iteration_total'] = int(match.group(3))
                parsed_data['consumed_samples'] = int(match.group(4))
                parsed_data['elapsed_time_per_iteration_ms'] = float(match.group(5))
                parsed_data['learning_rate'] = float(match.group(6))
                parsed_data['global_batch_size'] = int(match.group(7))
                parsed_data['lm_loss'] = float(match.group(8))
                parsed_data['loss_scale'] = float(match.group(9))
                parsed_data['grad_norm'] = float(match.group(10))
                parsed_data['skipped_iterations'] = int(match.group(11))
                parsed_data['nan_iterations'] = int(match.group(12))
            # Reset current_time_block after processing the main line, 
            # as time blocks typically follow
            current_time_block = None 
            continue

        # 3. Identify start of a new rank timing block (e.g., 'forward-backward:')
        time_block_match = time_block_start_pattern.match(line)
        if time_block_match and not line.startswith('times across ranks'):
            current_time_block = time_block_match.group(1).replace('-', '_') + '_times_ms'
            # Initialize as an empty dictionary if it doesn't exist
            if current_time_block not in parsed_data:
                parsed_data[current_time_block] = {}
            continue
        
        # 4. Parse rank-specific timing data (e.g., 'rank 0: 531.53')
        if current_time_block:
            rank_time_match = rank_time_pattern.match(line)
            if rank_time_match:
                rank = int(rank_time_match.group(1))
                time_ms = float(rank_time_match.group(2))
                parsed_data[current_time_block][rank] = time_ms
                continue

    return parsed_data

# --- New Wrapper Function to Read the File ---

def process_log_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a log file, splits it into individual log blocks, and parses each one.

    Args:
        file_path: The path to the log file.

    Returns:
        A list of dictionaries, where each dictionary represents a parsed log entry.
    """
    all_parsed_logs: List[Dict[str, Any]] = []
    
    # Define the pattern that marks the start of a new log block (the timestamped line)
    # The pattern checks for: start of line, optional spaces, [, four digits (year), -, etc.
    block_start_pattern = re.compile(r"mixtera average feedback time")

    try:
        with open(file_path, 'r') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []

    log_content_split = block_start_pattern.sub(r'###LOG_SPLIT###\g<0>', log_content)

    # Split the content by the placeholder and ignore the first empty element
    raw_blocks = log_content_split.split('###LOG_SPLIT###')[1:]
    
    # 2. Iterate and parse each raw block
    for block in raw_blocks:
        if block.strip(): # Ensure the block is not empty
            parsed_block = parse_training_log_block(block)
            if parsed_block and parsed_block['iteration_current'] > 100:
                all_parsed_logs.append(parsed_block)

    print(f"✅ Successfully parsed {len(all_parsed_logs)} log blocks.")
    return all_parsed_logs

# --- Example Usage ---

file_path = './experiments/162M-32node-ado-verbose/full.out'
try:
    # Process the created file
    results = process_log_file(file_path)

except Exception as e:
    print(f"An error occurred during file operation: {e}")
    
    
    
# Plotting
plt.figure(figsize=(12, 6))

# We will map each metric+type to a Y-coordinate
# 1: Batch Min, 2: Batch Max, 3: Forward Min, 4: Forward Max

# Filter data
mixtera_data = [r['mixtera'] for r in results]
forward_data = [r['forward_compute_times_ms'][0] for r in results]
backward_data = [r['backward_compute_times_ms'][0] for r in results]
batch_data = [r['batch_generator_times_ms'][0] for r in results]

jitter_strength = 0.08

# Helper to plot a strip
def plot_strip(values, y_center, color, label):
    y_noise = np.random.normal(y_center, jitter_strength, size=len(values))
    plt.scatter(values, y_noise, alpha=0.5, s=25, color=color, label=label, edgecolors='none')
# Plot Batch Generator (Green/Purple)
plot_strip(mixtera_data, 1, 'green', 'mixtera feedback')

plot_strip(forward_data, 2, 'orange', 'forward')

plot_strip(backward_data, 3, 'blue', 'backward')

plot_strip(batch_data, 4, 'red', 'batch generator')
# Formatting
plt.yticks([1, 2, 3, 4], [
    'mixtera\nfeedback', 
    'forward',
    'backward',
    'batch\ngenerator'
])
plt.xlabel('Time (ms)')
plt.title('Distribution of Latency (128 GPUs with ADO Mixing)')
plt.grid(True, axis='x', linestyle='--', alpha=0.5)

# Cleanup spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig('distribution_plot.png')
plt.show()