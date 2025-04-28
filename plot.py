import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Read the CSV file
plot_folder_name = 'run_20250426_230358'
csv_file_name = 'risk_stats_iter3.csv'
file_path = os.path.join(os.path.dirname(__file__), 'plots', plot_folder_name, csv_file_name)
df = pd.read_csv(file_path)

# Create a figure with a larger size
plt.figure(figsize=(12, 8))

# Plot multiple metrics
plt.plot(df['iteration'], df['risk_max_similarity'], label='Max Similarity', marker='o')
plt.plot(df['iteration'], df['risk_min_similarity'], label='Min Similarity', marker='s')
plt.plot(df['iteration'], df['risk_avg_similarity'], label='Avg Similarity', marker='^')

# Customize the plot
plt.title('Risk Similarities per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Similarity Score')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Generate timestamp with millisecond precision
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # [:-3] keeps milliseconds only

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, 'plots', 'post_generated_plots')

# Create the filename with timestamp
output_filename = os.path.join(output_dir, f'risk_similarities_{timestamp}.svg')

# Save the plot
plt.savefig(output_filename)

plt.close()
