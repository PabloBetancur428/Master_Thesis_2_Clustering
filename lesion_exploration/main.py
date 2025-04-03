"""
main.py

Orchestrates the lesion volume analysis pipeline:
1) Loads lesion volume data from the directory structure.
2) Aggregates the data into a DataFrame.
3) Visualizes the global distribution of lesion volumes.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_all_lesion_volumes
import visualization_vol as vl #import plot_vome_distribution
from save_explore_dataframe import save_and_explore_dataframe 

def main():
    # Define the root directories for baseline or follow-up data.
    # For example, change this path to the folder that contains patient folders.
    baseline_dir = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/clean_baseline"
    follow_up_dir = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/clean_follow_up"
    
    # Load lesion volume data from the folder structure.
    df = load_all_lesion_volumes(baseline_dir, follow_up_dir)
    
    if df.empty:
        print("No lesion volume data found.")
        return

    print("Lesion volume DataFrame:")
    print(df.head())
    
    # Visualize the overall distribution of lesion volumes.
    vl.plot_volume_distribution(df, output_path="lesion_volume_distribution.png")
    save_and_explore_dataframe(df, output_dir="lesion_exploration/output", filename_prefix="raw")

if __name__ == "__main__":
    main()