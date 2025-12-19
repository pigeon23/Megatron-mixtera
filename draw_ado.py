import json
import matplotlib.pyplot as plt

torchtitan = "/users/yiswang/scratch/torchtitan-mixtera/experiments/adolog/torchtitan_pile1k_1b_ado_default3_seqfix.json"
path = "./experiments/1B-default-ADO/adolog/Megatron-mixtera1_seqfix.json"

with open(torchtitan, 'r') as f:
    data = json.load(f)
    
name = {0: "ArXiv", 
        1: "BookCorpus2",
        2: "Books3",
        3: "DM Mathematics", 
        4: "Enron Emails", 
        5: "EuroParl", 
        6: "FreeLaw", 
        7: "Github", 
        8: "Gutenberg (PG-19)", 
        9: "HackerNews", 
        10: "NIH ExPorter", 
        11: "OpenSubtitles", 
        12: "OpenWebText2", 
        13: "PhilPapers", 
        14: "Pile-CC", 
        15: "PubMed Abstracts", 
        16: "PubMed Central", 
        17: "StackExchange", 
        18: "USPTO Backgrounds", 
        19: "Ubuntu IRC", 
        20: "Wikipedia (en)", 
        21: "YoutubeSubtitles"
}

inits = [0.1052, 0.0427, 0.1121, 0.0676, 0.1247, 0.1071]
l = [0, 7, 14, 2, 12, 16]

# 1. Store the data and labels
plot_data = []
length = len(data['entries'])

for i, init in zip(l, inits):
    x = []
    
    for j in range(length):
        if 'pi_t' in data['entries'][j]: 
            x.append(data['entries'][j]['pi_t'][i])
        else:
            x.append(init)
    
    label = name[i]
    # Store the list of y-values, the label, and the last value for sorting
    plot_data.append({'y_values': x, 'label': label, 'last_value': x[-1]})

# 2. Sort the data based on the 'last_value' (in descending order)
# Use reverse=True for descending order (highest last value first)
plot_data_sorted = sorted(plot_data, key=lambda item: item['last_value'], reverse=True)

# 3. Plot the data and collect the plot objects (lines) and labels in sorted order
lines = []
labels = []

plt.figure() # Create a new figure
x_range = range(length)

for item in plot_data_sorted:
    # Plot the line and capture the Line2D object
    line, = plt.plot(x_range, item['y_values'], label=item['label'])
    lines.append(line)
    labels.append(item['label'])

# 4. Apply title and legend using the sorted lines and labels
plt.title('Mixture Weights over Training Steps')

# Use the collected lines and labels for the legend
# The legend is created using the lines and labels in the sorted order
plt.legend(lines, labels) 
plt.savefig('./mixture_torchtitan.png')
