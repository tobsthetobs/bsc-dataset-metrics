import numpy as np

# function to calculate entropy of given array of data.
def calculate_shannon_entropy(data):
    N = len(data)
    counts = {}
    for i in data:
        if i not in counts:
            counts[i] = 0
        counts[i] += 1
    entropy = 0
    for count in counts.values():
        probability = count / N
        entropy -= probability * np.log2(probability)
    return entropy