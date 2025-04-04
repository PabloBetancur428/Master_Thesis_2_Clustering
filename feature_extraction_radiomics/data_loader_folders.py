"""
This module provides functions to traverse the directory structure
and extract the file paths for each patient.
The expected folder structure is:

root_dir/
 ├── Patient_001/
 │    ├── 2023/ or 2023_2/ etc.
 │    │    └── registered/
 │    │         ├── T1_corrected_canonical.nii_toMag.nii.gz
 │    │         ├── T2_FLAIR_corrected_canonical.nii_toMag.nii.gz
 │    │         ├── QSM_canonical.nii.gz
 │    │         └── lesions_MSpace_Mask.nii.gz"
 """


import os
import glob

def load_patient_files(root_dir, folder_type):
    """
    Traverse the directory structure for a given root directory.
    For each patient, find the first year folder (starting with "20")
    and the 'registered' subfolder inside it. Then, locate the files:
      - T1: "T1_corrected_canonical.nii_toMag.nii.gz"
      - T2: "T2_FLAIR_corrected_canonical.nii_toMag.nii.gz"
      - QSM: "QSM_canonical.nii.gz"
      - Mask: "lesions_MSpace_Mask.nii.gz"

    Parameters
    ----------
    root_dir : str
        Path to the root folder (e.g., baseline or follow-up).
    folder_type : str
        Label indicating the folder type ("baseline" or "follow_up").

    Returns
    -------
    list of dict
        Each dictionary contains:
          - PatientID
          - FolderType
          - Year (folder name)
          - T1, T2, QSM, Mask (file paths)
    """
    records = []
    for patient in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient)
        if not os.path.isdir(patient_path):
            continue

        # Iterate over year folders: names that start with "20"
        for year_folder in sorted(os.listdir(patient_path)):
            if not year_folder.startswith("20"):
                continue
            year_path = os.path.join(patient_path, year_folder)
            if not os.path.isdir(year_path):
                continue

            # Navigate to the "registered" folder inside the year folder
            registered_path = os.path.join(year_path, "registered")
            if not os.path.isdir(registered_path):
                continue

            # Define expected file paths using glob
            t1_list = glob.glob(os.path.join(registered_path, "T1_corrected_canonical.nii_toMag.nii.gz"))
            t2_list = glob.glob(os.path.join(registered_path, "T2_FLAIR_corrected_canonical.nii_toMag.nii.gz"))
            qsm_list = glob.glob(os.path.join(registered_path, "QSM_canonical.nii.gz"))
            mask_list = glob.glob(os.path.join(registered_path, "lesions_MSpace_Mask.nii.gz"))

            # Only add record if all files are found
            if t1_list and t2_list and qsm_list and mask_list:
                record = {
                    "PatientID": patient,
                    "FolderType": folder_type,
                    "Year": year_folder,
                    "T1": t1_list[0],
                    "T2": t2_list[0],
                    "QSM": qsm_list[0],
                    "Mask": mask_list[0]
                }
                records.append(record)
                # Only process the first year folder per patient
                break
    return records

def load_all_patient_files(baseline_dir, followup_dir):
    """
    Process both baseline and follow-up directories and combine their records.

    Parameters
    ----------
    baseline_dir : str
        Path to the baseline folder.
    followup_dir : str
        Path to the follow-up folder.

    Returns
    -------
    list of dict
        Combined records from both baseline and follow-up.
    """
    baseline_records = load_patient_files(baseline_dir, "baseline")
    followup_records = load_patient_files(followup_dir, "follow_up")
    return baseline_records + followup_records


if __name__ == "__main__":

    base_dir = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/clean_baseline"
    follow_dir = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/clean_follow_up"

    total_data = load_all_patient_files(base_dir, follow_dir)

    print(total_data)