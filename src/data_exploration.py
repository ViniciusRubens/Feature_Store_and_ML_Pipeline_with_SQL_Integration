import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

import matplotlib
matplotlib.use('Agg')

def plot_feature_distributions(features_df, save_path=None):
    """
    Plots histograms and saves to disk.
    """

    # Calculate number of numeric features (excluding target)
    num_features = len(features_df.columns) - 1
    cols = 3
    rows = (num_features + cols - 1) // cols

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, rows * 3))
    axes = axes.flatten()

    # Loop through features (excluding the last column which is target)
    for i, col in enumerate(features_df.columns[:-1]):
        sns.histplot(features_df[col], ax=axes[i], kde=True)
        axes[i].set_title(col)
        axes[i].set_ylabel('Count')

    # Hide unused axes
    for i in range(num_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    
    if save_path:
        full_path = os.path.join(save_path, 'feature_distributions.png')
        plt.savefig(full_path)
        print(f"   [Artifact Saved] Distributions plot saved to: {full_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_feature_correlations(features_df, save_path=None):
    """
    Plots correlation heatmap and saves to disk.
    """
    correlation_matrix = features_df.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Correlations')
    
    if save_path:
        full_path = os.path.join(save_path, 'feature_correlations.png')
        plt.savefig(full_path)
        print(f"   [Artifact Saved] Correlations plot saved to: {full_path}")
        plt.close()
    else:
        plt.show()


def analyze_data(features_df, artifacts_path='pipeline_runs'):
    """
    Orchestrates the data analysis process and saving of artifacts.
    """
    
    os.makedirs(artifacts_path, exist_ok=True)

    print("\n--- Generating Feature Distributions Artifact ---")
    plot_feature_distributions(features_df, save_path=artifacts_path)

    print("\n--- Generating Feature Correlations Artifact ---")
    plot_feature_correlations(features_df, save_path=artifacts_path)