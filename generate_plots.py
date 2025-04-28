import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Set font sizes
xlabel_fontsize = 12
ylabel_fontsize = 12
title_fontsize = 14
tick_fontsize = 10
legend_fontsize = 10

# Read the CSV
plot_folder_name = 'run_20250426_230358'
csv_file_name = 'risk_stats_iter3.csv'
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'plots', plot_folder_name, csv_file_name)
df = pd.read_csv(file_path)

# Output folder setup
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Up to milliseconds
run_output_dir = os.path.join(current_dir, 'plots', 'post_generated_plots')
os.makedirs(run_output_dir, exist_ok=True)

# Get last iteration number
last_iteration = df['iteration'].iloc[-1]

# --- Plot 1: Impact Statistics ---
plt.figure(figsize=(10, 6), dpi=1200)
plt.plot(df['iteration'], df['impact_mean'], 'g-o', label='Mean Impact')
plt.fill_between(df['iteration'], df['impact_mean'] - df['impact_std'], df['impact_mean'] + df['impact_std'], alpha=0.2, color='g', label='±1 Std Dev')
plt.plot(df['iteration'], df['impact_max'], 'r--', label='Max Impact')
plt.plot(df['iteration'], df['impact_min'], 'b--', label='Min Impact')
plt.xlabel('Iteration', fontsize=xlabel_fontsize)
plt.ylabel('Impact Level', fontsize=ylabel_fontsize)
plt.title('Impact Statistics Across Iterations', fontsize=title_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.grid(True)
plt.legend(fontsize=legend_fontsize)
filename1 = os.path.join(run_output_dir, f"impact_stats_iter{last_iteration}.svg")
plt.savefig(filename1, dpi=1200, bbox_inches='tight')
plt.close()

# --- Plot 1.1: Impact Confidence Statistics ---
plt.figure(figsize=(10, 6), dpi=1200)
plt.plot(df['iteration'], df['impact_confidence_mean'], 'g-o', label='Mean Impact Confidence')
plt.fill_between(df['iteration'], df['impact_confidence_mean'] - df['impact_confidence_std'], df['impact_confidence_mean'] + df['impact_confidence_std'], alpha=0.2, color='g', label='±1 Std Dev')
plt.plot(df['iteration'], df['impact_confidence_max'], 'r--', label='Max Impact Confidence')
plt.plot(df['iteration'], df['impact_confidence_min'], 'b--', label='Min Impact Confidence')
plt.xlabel('Iteration', fontsize=xlabel_fontsize)
plt.ylabel('Impact Confidence', fontsize=ylabel_fontsize)
plt.title('Impact Confidence Statistics Across Iterations', fontsize=title_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.grid(True)
plt.legend(fontsize=legend_fontsize)
filename1_1 = os.path.join(run_output_dir, f"impact_confidence_stats_iter{last_iteration}.svg")
plt.savefig(filename1_1, dpi=1200, bbox_inches='tight')
plt.close()

# --- Plot 2: Number of Risks per Iteration ---
plt.figure(figsize=(10, 6), dpi=1200)
plt.plot(df['iteration'], df['num_risks'], 'b-o')
plt.xlabel('Iteration', fontsize=xlabel_fontsize)
plt.ylabel('Number of Risks', fontsize=ylabel_fontsize)
plt.title('Number of Risks per Iteration', fontsize=title_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.grid(True)
filename2 = os.path.join(run_output_dir, f"num_risks_iter{last_iteration}.svg")
plt.savefig(filename2, dpi=1200, bbox_inches='tight')
plt.close()

# --- Plot 3: Likelihood Statistics ---
plt.figure(figsize=(10, 6), dpi=1200)
plt.plot(df['iteration'], df['likelihood_mean'], 'g-o', label='Mean Likelihood')
plt.fill_between(df['iteration'], df['likelihood_mean'] - df['likelihood_std'], df['likelihood_mean'] + df['likelihood_std'], alpha=0.2, color='g', label='±1 Std Dev')
plt.plot(df['iteration'], df['likelihood_max'], 'r--', label='Max Likelihood')
plt.plot(df['iteration'], df['likelihood_min'], 'b--', label='Min Likelihood')
plt.xlabel('Iteration', fontsize=xlabel_fontsize)
plt.ylabel('Likelihood Level', fontsize=ylabel_fontsize)
plt.title('Likelihood Statistics Across Iterations', fontsize=title_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.grid(True)
plt.legend(fontsize=legend_fontsize)
filename3 = os.path.join(run_output_dir, f"likelihood_stats_iter{last_iteration}.svg")
plt.savefig(filename3, dpi=1200, bbox_inches='tight')
plt.close()

# --- Plot 3.1: Likelihood Confidence Statistics ---
plt.figure(figsize=(10, 6), dpi=1200)
plt.plot(df['iteration'], df['likelihood_confidence_mean'], 'g-o', label='Mean Likelihood Confidence')
plt.fill_between(df['iteration'], df['likelihood_confidence_mean'] - df['likelihood_confidence_std'], df['likelihood_confidence_mean'] + df['likelihood_confidence_std'], alpha=0.2, color='g', label='±1 Std Dev')
plt.plot(df['iteration'], df['likelihood_confidence_max'], 'r--', label='Max Likelihood Confidence')
plt.plot(df['iteration'], df['likelihood_confidence_min'], 'b--', label='Min Likelihood Confidence')
plt.xlabel('Iteration', fontsize=xlabel_fontsize)
plt.ylabel('Likelihood Confidence', fontsize=ylabel_fontsize)
plt.title('Likelihood Confidence Statistics Across Iterations', fontsize=title_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.grid(True)
plt.legend(fontsize=legend_fontsize)
filename3_1 = os.path.join(run_output_dir, f"likelihood_confidence_stats_iter{last_iteration}.svg")
plt.savefig(filename3_1, dpi=1200, bbox_inches='tight')
plt.close()

# (continues for all your other plots: combined impact & likelihood, semantic similarity, semantic difference, Jaccard similarity, risk similarity, average drift, etc.)

# --- Plot 4: Combined Impact and Likelihood Trends ---
plt.figure(figsize=(12, 6), dpi=1200)
plt.plot(df['iteration'], df['impact_mean'], 'b-o', label='Mean Impact')
plt.plot(df['iteration'], df['likelihood_mean'], 'r-o', label='Mean Likelihood')
plt.fill_between(df['iteration'], df['impact_mean'] - df['impact_std'], df['impact_mean'] + df['impact_std'], alpha=0.2, color='b', label='Impact ±1 Std Dev')
plt.fill_between(df['iteration'], df['likelihood_mean'] - df['likelihood_std'], df['likelihood_mean'] + df['likelihood_std'], alpha=0.2, color='r', label='Likelihood ±1 Std Dev')
plt.xlabel('Iteration', fontsize=xlabel_fontsize)
plt.ylabel('Level', fontsize=ylabel_fontsize)
plt.title('Impact and Likelihood Trends Across Iterations', fontsize=title_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.grid(True)
plt.legend(fontsize=legend_fontsize)
filename4 = os.path.join(run_output_dir, f"combined_impact_likelihood_iter{last_iteration}.svg")
plt.savefig(filename4, dpi=1200, bbox_inches='tight')
plt.close()

# --- Plot 5: Semantic Similarity Metrics ---
plt.figure(figsize=(12, 6), dpi=1200)
plt.plot(df['iteration'], df['similarity_score'], 'b-o', label='Semantic Similarity Score')
plt.xlabel('Iteration', fontsize=xlabel_fontsize)
plt.ylabel('Cosine Similarity Score', fontsize=ylabel_fontsize)
plt.title('Semantic Similarity Metric Across Iterations', fontsize=title_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.grid(True)
plt.legend(fontsize=legend_fontsize)
filename5 = os.path.join(run_output_dir, f"semantic_similarity_metrics_iter{last_iteration}.svg")
plt.savefig(filename5, dpi=1200, bbox_inches='tight')
plt.close()

# --- Plot 6: Semantic Difference Metrics ---
plt.figure(figsize=(12, 6), dpi=1200)
plt.plot(df['iteration'], df['semantic_difference'], 'r-o', label='Semantic Difference')
plt.xlabel('Iteration', fontsize=xlabel_fontsize)
plt.ylabel('Cosine Distance', fontsize=ylabel_fontsize)
plt.title('Semantic Difference Metric Across Iterations', fontsize=title_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.grid(True)
plt.legend(fontsize=legend_fontsize)
filename6 = os.path.join(run_output_dir, f"semantic_difference_metrics_iter{last_iteration}.svg")
plt.savefig(filename6, dpi=1200, bbox_inches='tight')
plt.close()

# --- Plot 7: Jaccard Similarity Metrics ---
plt.figure(figsize=(12, 6), dpi=1200)
plt.plot(df['iteration'], df['jaccard_similarity'], 'g-o', label='Jaccard Similarity')
plt.xlabel('Iteration', fontsize=xlabel_fontsize)
plt.ylabel('Jaccard Score', fontsize=ylabel_fontsize)
plt.title('Jaccard Similarity Metrics Across Iterations', fontsize=title_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.grid(True)
plt.legend(fontsize=legend_fontsize)
filename7 = os.path.join(run_output_dir, f"jaccard_similarity_metrics_iter{last_iteration}.svg")
plt.savefig(filename7, dpi=1200, bbox_inches='tight')
plt.close()

# --- Plot 8: Risk Similarity Metrics ---
plt.figure(figsize=(12, 6), dpi=1200)
plt.plot(df['iteration'], df['risk_avg_similarity'], 'b-o', label='Avg Risk Similarity')
plt.plot(df['iteration'], df['risk_max_similarity'], 'r--', label='Max Risk Similarity')
plt.plot(df['iteration'], df['risk_min_similarity'], 'g--', label='Min Risk Similarity')
plt.xlabel('Iteration', fontsize=xlabel_fontsize)
plt.ylabel('Risks Cosine Similarity Score', fontsize=ylabel_fontsize)
plt.title('Risk Similarity Metrics Across Iterations', fontsize=title_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.grid(True)
plt.legend(fontsize=legend_fontsize)
filename8 = os.path.join(run_output_dir, f"risk_similarity_metrics_iter{last_iteration}.svg")
plt.savefig(filename8, dpi=1200, bbox_inches='tight')
plt.close()

# --- Plot 9: Average Risk Drift ---
plt.figure(figsize=(12, 6), dpi=1200)
plt.plot(df['iteration'], df['risk_avg_risk_drift'], 'b-o', label='Average Risk Drift')
plt.xlabel('Iteration', fontsize=xlabel_fontsize)
plt.ylabel('Average Risk Drift', fontsize=ylabel_fontsize)
plt.title('Average Risk Drift Across Iterations', fontsize=title_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.grid(True)
plt.legend(fontsize=legend_fontsize)
filename9 = os.path.join(run_output_dir, f"avg_risk_drift_iter{last_iteration}.svg")
plt.savefig(filename9, dpi=1200, bbox_inches='tight')
plt.close()

# --- Plot 10: Proportion of Risks Above Threshold ---
plt.figure(figsize=(12, 6), dpi=1200)
plt.plot(df['iteration'], df['proportion_above_threshold'], 'b-o', label='Proportion of Risks Above Threshold')
plt.xlabel('Iteration', fontsize=xlabel_fontsize)
plt.ylabel('Proportion', fontsize=ylabel_fontsize)
plt.title('Proportion of Risks Above Threshold Across Iterations', fontsize=title_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.grid(True)
plt.legend(fontsize=legend_fontsize)
filename10 = os.path.join(run_output_dir, f"proportion_above_threshold_iter{last_iteration}.svg")
plt.savefig(filename10, dpi=1200, bbox_inches='tight')
plt.close()

