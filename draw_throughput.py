import re
import numpy as np
import matplotlib.pyplot as plt

def parse_training_log(line):
    # Regex pattern using named groups for clarity
    w_tp_pattern = (
        r"\[(?P<timestamp>.*?)\]\s+iteration\s+(?P<iteration>\d+)/\s+(?P<total_iterations>\d+)\s+\|\s+"
        r"consumed samples:\s+(?P<samples>\d+)\s+\|\s+"
        r"elapsed time per iteration \(ms\):\s+(?P<ms_per_iter>[\d.]+)\s+\|\s+"
        r"throughput per GPU \(TFLOP/s/GPU\):\s+(?P<tflops>[\d.]+)\s+\|\s+"
        r"learning rate:\s+(?P<lr>[\d.E+-]+)\s+\|\s+"
        r"global batch size:\s+(?P<batch_size>\d+)\s+\|\s+"
        r"lm loss:\s+(?P<loss>[\d.E+-]+)\s+\|\s+"
        r"loss scale:\s+(?P<loss_scale>[\d.]+)\s+\|\s+"
        r"grad norm:\s+(?P<grad_norm>[\d.]+)\s+\|\s+"
        r"number of skipped iterations:\s+(?P<skipped>\d+)\s+\|\s+"
        r"number of nan iterations:\s+(?P<nan_iters>\d+)\s+\|"
    )
    
    wo_tp_pattern = (
        r"\[(?P<timestamp>.*?)\]\s+iteration\s+(?P<iteration>\d+)/\s+(?P<total_iterations>\d+)\s+\|\s+"
        r"consumed samples:\s+(?P<samples>\d+)\s+\|\s+"
        r"elapsed time per iteration \(ms\):\s+(?P<ms_per_iter>[\d.]+)\s+\|\s+"
        r"learning rate:\s+(?P<lr>[\d.E+-]+)\s+\|\s+"
        r"global batch size:\s+(?P<batch_size>\d+)\s+\|\s+"
        r"lm loss:\s+(?P<loss>[\d.E+-]+)\s+\|\s+"
        r"loss scale:\s+(?P<loss_scale>[\d.]+)\s+\|\s+"
        r"grad norm:\s+(?P<grad_norm>[\d.]+)\s+\|\s+"
        r"number of skipped iterations:\s+(?P<skipped>\d+)\s+\|\s+"
        r"number of nan iterations:\s+(?P<nan_iters>\d+)\s+\|"
    )

    match = re.search(wo_tp_pattern, line)
    if match:
        data = match.groupdict()
        
        # Convert numeric strings to appropriate types
        data['iteration'] = int(data['iteration'])
        data['total_iterations'] = int(data['total_iterations'])
        data['samples'] = int(data['samples'])
        data['ms_per_iter'] = float(data['ms_per_iter'])
        data['lr'] = float(data['lr'])
        data['batch_size'] = int(data['batch_size'])
        data['loss'] = float(data['loss'])
        data['loss_scale'] = float(data['loss_scale'])
        data['grad_norm'] = float(data['grad_norm'])
        data['skipped'] = int(data['skipped'])
        data['nan_iters'] = int(data['nan_iters'])
        
        return data
    
    match = re.search(w_tp_pattern, line)
    if match:
        data = match.groupdict()
        
        # Convert numeric strings to appropriate types
        data['iteration'] = int(data['iteration'])
        data['total_iterations'] = int(data['total_iterations'])
        data['samples'] = int(data['samples'])
        data['ms_per_iter'] = float(data['ms_per_iter'])
        data['tflops'] = float(data['tflops'])
        data['lr'] = float(data['lr'])
        data['batch_size'] = int(data['batch_size'])
        data['loss'] = float(data['loss'])
        data['loss_scale'] = float(data['loss_scale'])
        data['grad_norm'] = float(data['grad_norm'])
        data['skipped'] = int(data['skipped'])
        data['nan_iters'] = int(data['nan_iters'])
        
        return data
    return None

nodes = [1, 2, 4, 8, 16, 32]
tp = {}

for node in nodes:
    tp[node] = []
    with open(f'./experiments/162M-{node}node/clean.out', 'r') as f:
        for log_line in f:
            try:
                log_line = log_line.strip()
                # Execute and print
                parsed_data = parse_training_log(log_line)
                iteration_time = parsed_data['ms_per_iter'] / 1000  # Convert ms to seconds
                throughput = 2048 * parsed_data['batch_size'] / iteration_time / 2**20
                tp[node].append(throughput)
            except Exception as e:
                print(f"Error parsing node {node} line: {log_line}\n{e}")
    tp[node] = tp[node][1:]  
            
mu = [np.mean(l) for l in tp.values()]
sigma = [np.std(l)  for l in tp.values()]

rects = plt.bar([np.log2(n) for n in nodes], mu, yerr=sigma, color='green', capsize=5)
plt.xlabel('Number of GPUs')
plt.ylabel('Throughput (MTokens/s)')
plt.xticks([np.log2(n) for n in nodes], [str(n * 4) for n in nodes])
plt.title('Throughput vs Number of GPUs')
plt.savefig('throughput.png')