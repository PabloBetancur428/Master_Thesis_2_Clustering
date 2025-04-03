"""
data_loader.py

Module for loading lesion volume data from a directory structure.
Iterate through patient directories, go to the year, find the folder "registered",
and locate the file "lesions_MSpace_Mask.nii"

"""

import os
import pandas as pd
from volume_extractor import LesionVolumeExtractor  

def load_lesion_volumes_from_root(root_dir, folder_type_label):
    """
    Iterate over patient folders in root_dir, extract lesion volumes from the registered year folder,
    and return a list of records with patient ID, folder type, year folder, lesion label, and volume.

    Parameters
    ----------
    root_dir : str
        Root directory containing patient folders.
    folder_type_label : str
        A label to indicate the folder type (e.g., 'baseline' or 'follow_up').

    Returns
    -------
    list of dict
        Each dict contains: PatientID, FolderType, YearFolder, LesionLabel, LesionVolume.
    """
    records = []
    for patient_id in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_id)
        if not os.path.isdir(patient_path):
            continue

        # Iterate over year folders (folder name starts with "20")
        for year_folder in os.listdir(patient_path):
            if not year_folder.startswith("20"):
                continue
            year_path = os.path.join(patient_path, year_folder)
            if not os.path.isdir(year_path):
                continue

            # Navigate to the "registered" folder inside the year folder
            registered_path = os.path.join(year_path, "registered")
            if not os.path.isdir(registered_path):
                continue

            # Look for the lesion mask file in the "registered" folder.
            lesion_mask_file = None
            for fname in os.listdir(registered_path):
                if "lesions_MSpace_Mask.nii.gz" in fname and fname.endswith(".nii.gz"):
                    lesion_mask_file = os.path.join(registered_path, fname)
                    break
            if lesion_mask_file is None:
                continue

            # Use the LesionVolumeExtractor to extract volumes from this mask.
            extractor = LesionVolumeExtractor(lesion_mask_file)
            extractor.load_mask()
            lesion_volumes = extractor.extract_lesion_volumes()

            # For each lesion label, create a record
            for lesion_label, volume in lesion_volumes.items():
                record = {
                    "PatientID": patient_id,
                    "FolderType": folder_type_label,
                    "YearFolder": year_folder,
                    "LesionLabel": lesion_label,
                    "LesionVolume": volume
                }
                records.append(record)
    return records

def load_all_lesion_volumes(baseline_dir, followup_dir):
    """
    Process both baseline and follow-up directories and combine their lesion volume data
    into a single DataFrame.
    
    Parameters
    ----------
    baseline_dir : str
    followup_dir : str
    
    Returns
    -------
    pd.DataFrame
    """
    baseline_records = load_lesion_volumes_from_root(baseline_dir, folder_type_label="baseline")
    followup_records = load_lesion_volumes_from_root(followup_dir, folder_type_label="follow_up")
    all_records = baseline_records + followup_records
    df = pd.DataFrame(all_records)
    return df


