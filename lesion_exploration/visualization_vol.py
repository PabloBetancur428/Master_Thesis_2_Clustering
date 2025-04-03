"""
visualization.py

Module for visualizing the distribution of lesion volumes.
Provides functions to create global plots (e.g., histogram, box plot, violin plot)
from a DataFrame containing lesion volume data.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_volume_distribution(df, output_path=None):
    """
    Create a global plot of lesion volumes across all patients.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: PatientID, YearFolder, LesionLabel, LesionVolume.
    output_path : str, optional
        If provided, the plot is saved to this file.
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(y=df["LesionVolume"], inner="quartile", color="skyblue")
    plt.title("Distribution of Lesion Volumes Across Patients")
    plt.ylabel("Lesion Volume (physical units)")
    plt.xlabel("All Lesions")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()