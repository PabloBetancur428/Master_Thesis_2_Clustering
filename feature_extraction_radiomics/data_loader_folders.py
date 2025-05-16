import os
import glob
import pandas as pd

def load_patient_files(root_dir, folder_type, patient_set):
    """
    Traverse the directory structure under root_dir for the patients in patient_set.
    For each patient in patient_set, find the first year folder (starting with "20")
    and its 'registered' subfolder, then locate the files:
      - T1: "T1_corrected_canonical.nii_toMag.nii.gz"
      - T2: "T2_FLAIR_corrected_canonical.nii_toMag.nii.gz"
      - QSM: "QSM_canonical.nii.gz"
      - Mask: "lesions_MSpace_Mask.nii.gz"

    Parameters
    ----------
    root_dir : str
        Path to the root folder (e.g., baseline or follow-up).
    folder_type : str
        Label indicating the folder type ("baseline" or "follow_up"), 
        used to populate the record.
    patient_set : set of str
        PatientID strings to include for this folder_type.

    Returns
    -------
    list of dict
        Each dict contains:
          - PatientID, FolderType, Year, T1, T2, QSM, Mask (file paths)
    """
    records = []
    for patient in os.listdir(root_dir):
        # only keep patients in our include list
        if patient not in patient_set:
            continue

        patient_path = os.path.join(root_dir, patient)
        if not os.path.isdir(patient_path):
            continue

        # iterate sorted year folders starting with "20"
        for year_folder in sorted(os.listdir(patient_path)):
            if not year_folder.startswith("20"):
                continue
            registered_path = os.path.join(patient_path, year_folder, "registered")
            if not os.path.isdir(registered_path):
                continue

            # look for the four required files
            t1_list   = glob.glob(os.path.join(registered_path, "T1_corrected_canonical.nii_toMag.nii.gz"))
            t2_list   = glob.glob(os.path.join(registered_path, "T2_FLAIR_corrected_canonical.nii_toMag.nii.gz"))
            qsm_list  = glob.glob(os.path.join(registered_path, "QSM_canonical.nii.gz"))
            mask_list = glob.glob(os.path.join(registered_path, "lesions_MSpace_Mask.nii.gz"))

            if t1_list and t2_list and qsm_list and mask_list:
                records.append({
                    "PatientID":  patient,
                    "FolderType": folder_type,
                    "Year":       year_folder,
                    "T1":         t1_list[0],
                    "T2":         t2_list[0],
                    "QSM":        qsm_list[0],
                    "Mask":       mask_list[0],
                })
                break  # only first valid year per patient
    return records

def load_all_patient_files(baseline_dir, followup_dir, excel_path):
    """
    Read excel_path to get which PatientIDs to include for baseline vs follow-up,
    then scan each directory accordingly.

    Returns a combined list of record dicts.
    """
    # Load include list
    df = pd.read_excel(excel_path, dtype=str)
    df = df.rename(columns=lambda c: c.strip())  # in case of extra spaces
    # Build sets
    baseline_set = set(df.loc[df.FolderType == "baseline", "PatientID"])
    followup_set = set(df.loc[df.FolderType == "follow_up", "PatientID"])

    # Scan directories, filtering by those sets
    baseline_records = load_patient_files(baseline_dir,  "baseline",  baseline_set)
    followup_records = load_patient_files(followup_dir, "follow_up", followup_set)
    return baseline_records + followup_records

if __name__ == "__main__":
    base_dir = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/clean_baseline"
    follow_dir = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/clean_follow_up"
    excel_file = "/home/jbetancur/Desktop/codes/clustering/patients_with_qsm.xlsx"

    total_data = load_all_patient_files(base_dir, follow_dir, excel_file)
    print(total_data)

