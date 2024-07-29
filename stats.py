import os
import re
import numpy as np

def find_statistics(root_dir):
    averages = []

    for subdir, _, files in os.walk(root_dir):
        if 'config_log.txt' in files:
            with open(os.path.join(subdir, 'config_log.txt'), 'r') as file:
                lines = file.readlines()
                last_line = lines[-3]
                match = re.search(r'Average: (\d+(\.\d+)?)', last_line)
                if match:
                    averages.append(float(match.group(1)))

    if averages:
        avg = np.mean(averages)
        std = np.std(averages)
        max_val = np.max(averages)
        min_val = np.min(averages)
        return round(avg, 2), round(std, 2), round(max_val, 2), round(min_val, 2)
    else:
        return None, None, None, None

root_dir = '/home/nhiguera/Research/KR2024/logs/SR/50/GPT3'
average, std_dev, max_val, min_val = find_statistics(root_dir)

if average is not None and std_dev is not None:
    print(f"Average of all X numbers: {average:.2f}")
    print(f"Standard Deviation of all X numbers: {std_dev:.2f}")
    print(f"Maximum of all X numbers: {max_val:.2f}")
    print(f"Minimum of all X numbers: {min_val:.2f}")
else:
    print("No valid 'Average: X' entries found in config_log.txt files.")
