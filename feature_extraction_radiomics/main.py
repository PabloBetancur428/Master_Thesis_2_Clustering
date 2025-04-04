"""Orchestrates the radiomic feature extraction and analysis pipeline across baseline and follow-up data.
For each patient record (found via data_loader), it:
  - Retrieves the "registered" folder from the expected file paths.
  - Uses RadiomicFeatureExtractor to load the images and extract 3D lesion features (each lesion is treated independently).
  - Augments each lesion record with PatientID, FolderType, and Year.
  - Aggregates all lesion-level features into a Pandas DataFrame for further analysis, clustering, and visualization.
"""

import os
import pandas as pd
from data_loader_folders import load_all_patient_files
from feature_extraction import RadiomicFeatureExtractor
# Optional: import clustering and visualization modules if needed
#from clustering import KMeansCluster
#from visualization import plot_volume_distribution

def main():
    # Define root directories for baseline and follow-up data.
    baseline_dir = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/clean_baseline"
    followup_dir = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/clean_follow_up"

    # Load patient records (each record contains file paths for T1, T2, QSM, Mask)
    patient_records = load_all_patient_files(baseline_dir, followup_dir)
    if not patient_records:
        print("No valid patient records found.")
        return

    # Global list to store lesion-level features across all patients
    all_lesion_features = []

    # Process each patient record
    for rec in patient_records:
        patient_id = rec["PatientID"]
        folder_type = rec["FolderType"]
        year_folder = rec["Year"]

        # Determine the registered folder (assumes T1 file path is in the registered folder)
        registered_folder = os.path.dirname(rec["T1"])
        print(f"Processing Patient {patient_id} ({folder_type}, {year_folder}) in {registered_folder}")

        try:
            extractor = RadiomicFeatureExtractor(registered_folder)
            extractor.load_images()
            lesion_feats = extractor.extract_features()  # List of dicts, one per lesion
        except Exception as e:
            print(f"Error processing patient {patient_id} in {year_folder}: {e}")
            continue

        # Add patient metadata to each lesion's features and append to global list
        for feat in lesion_feats:
            feat["PatientID"] = patient_id
            feat["FolderType"] = folder_type
            feat["YearFolder"] = year_folder
            all_lesion_features.append(feat)

    # Create a DataFrame from all lesion features
    df = pd.DataFrame(all_lesion_features)
    print("Aggregated Lesion Features DataFrame:")
    print(df.head())
    print("DataFrame shape:", df.shape) 
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join("feature_extraction", "output"), exist_ok=True)

    output_csv_path = os.path.join("feature_extraction", "output", "aggregated_lesion_features.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"Dataframe saved to {output_csv_path}")


    # (Optional) Save and visualize the results
    # For example, call your previously defined function to save/explore the DataFrame:
    # from save_explore_dataframe import save_and_explore_dataframe
    # save_and_explore_dataframe(df, output_dir="lesion_exploration/output")
    
    # (Optional) Proceed with clustering or further statistical analysis
    # Example: clustering with KMeansCluster, then visualize with PCA-based plots

if __name__ == "__main__":
    main()